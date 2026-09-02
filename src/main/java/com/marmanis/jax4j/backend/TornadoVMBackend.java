package com.marmanis.jax4j.backend;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Dispatches elementwise primitives to a specific TornadoVM {@link Device} via
 * {@code TornadoExecutionPlan.withDevice(...)}. Falls back to {@link HostBackend}
 * on any failure (no configured TornadoVM runtime, unsupported hardware, etc),
 * mirroring the defensive pattern already used by
 * {@code com.marmanis.jax4j.examples.FFIExamples#runCudaAdd}.
 *
 * <p>Only the elementwise ops have kernels here (ADD/SUB/MUL/DIV/EXP/LOG/SIN/COS).
 * DOT/SUM/MEAN need TornadoVM's reduction-specific annotations and are left as a
 * follow-up; {@code ConcreteNDArray} never routes them through this backend.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class TornadoVMBackend implements ExecutionBackend {
    private static final Logger log = LoggerFactory.getLogger(TornadoVMBackend.class);
    public static final TornadoVMBackend INSTANCE = new TornadoVMBackend();

    // ----------------------------------------------------------------
    // GAP-3: TaskGraph plan cache
    //
    // Key: (primitive, array-size, isF64, device).
    // Cached value: a pool of pre-built plans together with their bound float[]
    // buffers so we can copy fresh input data in before re-executing.
    // The TornadoVM transferToDevice(EVERY_EXECUTION, ...) mode reads
    // the array reference it was given at graph-construction time on
    // every execute() call, so writing new values into those arrays and
    // calling execute() again is the correct way to reuse a plan.
    // We pool plans to allow concurrent executions (e.g. inside pmap)
    // to borrow distinct buffers and plans without data races.
    // ----------------------------------------------------------------

    private record CacheKey(Primitive primitive, int size, int m, int k, int n, boolean isF64, Device device) {}

    private record CachedPlan(TornadoExecutionPlan plan,
                               float[] inA, float[] inB, float[] outF,
                               double[] inAd, double[] inBd, double[] outD) {}

    private static final java.util.concurrent.ConcurrentHashMap<CacheKey, java.util.concurrent.ConcurrentLinkedQueue<CachedPlan>> PLAN_CACHE =
        new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Maximum plans kept per (primitive, shape, dtype, device) cache key. Bounds the
     * memory a burst of concurrent calls (e.g. pmap over D devices) can wedge into
     * the cache; excess plans are closed rather than pooled. The size() check races
     * under contention so the pool may grow slightly above the bound, but not
     * unboundedly.
     */
    private static final int MAX_POOL_SIZE = 4;

    private static void boundedOffer(java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool, CachedPlan plan) {
        if (pool.size() < MAX_POOL_SIZE) {
            pool.offer(plan);
        } else {
            try { plan.plan().close(); } catch (Throwable ignore) {}
        }
    }

    @Override
    public float[] binary(Primitive primitive, float[] a, float[] b, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, false, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                System.arraycopy(b, 0, cached.inB(), 0, b.length);
                cached.plan().execute();
                float[] out = new float[a.length];
                System.arraycopy(cached.outF(), 0, out, 0, a.length);
                boundedOffer(pool, cached); // return to pool on success
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.binary(primitive, a, b, device);
            }
        }

        // Cache miss: build, execute.
        float[] inA = a.clone();
        float[] inB = b.clone();
        float[] out = new float[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inA, inB);
            tg = switch (primitive) {
                case ADD -> tg.task("k", TornadoVMBackend::vectorAdd, inA, inB, out);
                case SUB -> tg.task("k", TornadoVMBackend::vectorSub, inA, inB, out);
                case MUL -> tg.task("k", TornadoVMBackend::vectorMul, inA, inB, out);
                case DIV -> tg.task("k", TornadoVMBackend::vectorDiv, inA, inB, out);
                case MAX -> tg.task("k", TornadoVMBackend::vectorMax, inA, inB, out);
                case MIN -> tg.task("k", TornadoVMBackend::vectorMin, inA, inB, out);
                default -> throw new UnsupportedOperationException("No TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();
            
            // Store in pool (don't close the plan — it must stay alive for reuse).
            CachedPlan newPlan = new CachedPlan(plan, inA, inB, out, null, null, null);
            boundedOffer(pool, newPlan);

            float[] result = new float[a.length];
            System.arraycopy(out, 0, result, 0, a.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.binary(primitive, a, b, device);
        }
    }

    /**
     * Builds a {@link TornadoExecutionPlan} pinned to {@code device}, if any.
     * Callers must close the returned plan ({@code try}-with-resources) once
     * it's executed: closing frees the device buffers TornadoVM allocated for
     * this plan, and skipping it leaks GPU/OpenCL memory that accumulates
     * across every primitive call (every {@code ADD}/{@code EXP}/{@code DOT}
     * etc. builds a fresh {@link TaskGraph} and plan; none of this is cached
     * or reused across calls).
     */
    private static TornadoExecutionPlan configuredPlan(ImmutableTaskGraph itg, Device device) {
        TornadoExecutionPlan plan = new TornadoExecutionPlan(itg);
        return device.getTornadoDevice() != null ? plan.withDevice(device.getTornadoDevice()) : plan;
    }

    @Override
    public float[] unary(Primitive primitive, float[] a, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, false, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                cached.plan().execute();
                float[] out = new float[a.length];
                System.arraycopy(cached.outF(), 0, out, 0, a.length);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.unary(primitive, a, device);
            }
        }

        float[] inA = a.clone();
        float[] out = new float[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inA);
            tg = switch (primitive) {
                case EXP -> tg.task("k", TornadoVMBackend::vectorExp, inA, out);
                case LOG -> tg.task("k", TornadoVMBackend::vectorLog, inA, out);
                case SIN -> tg.task("k", TornadoVMBackend::vectorSin, inA, out);
                case COS -> tg.task("k", TornadoVMBackend::vectorCos, inA, out);
                case TANH -> tg.task("k", TornadoVMBackend::vectorTanh, inA, out);
                case RELU -> tg.task("k", TornadoVMBackend::vectorRelu, inA, out);
                case SIGMOID -> tg.task("k", TornadoVMBackend::vectorSigmoid, inA, out);
                case SQRT -> tg.task("k", TornadoVMBackend::vectorSqrt, inA, out);
                case RSQRT -> tg.task("k", TornadoVMBackend::vectorRsqrt, inA, out);
                default -> throw new UnsupportedOperationException("No TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, inA, null, out, null, null, null);
            boundedOffer(pool, newPlan);

            float[] result = new float[a.length];
            System.arraycopy(out, 0, result, 0, a.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.unary(primitive, a, device);
        }
    }

    @Override
    public float[] reduce(Primitive primitive, float[] a, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, false, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                cached.outF()[0] = 0f; // reset accumulator for TornadoVM @Reduce
                cached.plan().execute();
                float[] out = new float[1];
                out[0] = cached.outF()[0];
                if (primitive == Primitive.MEAN) {
                    out[0] /= a.length;
                }
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.reduce(primitive, a, device);
            }
        }

        float[] inA = a.clone();
        float[] out = new float[1];
        try {
            TaskGraph tg = new TaskGraph("jax4j_reduce_" + primitive + "_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inA);
            tg = switch (primitive) {
                case SUM, MEAN -> tg.task("k", TornadoVMBackend::reduceSum, inA, out);
                default -> throw new UnsupportedOperationException("No TornadoVM reduction kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, inA, null, out, null, null, null);
            boundedOffer(pool, newPlan);

            float[] result = new float[1];
            result[0] = out[0];
            if (primitive == Primitive.MEAN) {
                result[0] /= a.length;
            }
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.reduce(primitive, a, device);
        }
    }

    @Override
    public double[] reduce(Primitive primitive, double[] a, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, true, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inAd(), 0, a.length);
                cached.outD()[0] = 0.0; // reset accumulator for TornadoVM @Reduce
                cached.plan().execute();
                double[] out = new double[1];
                out[0] = cached.outD()[0];
                if (primitive == Primitive.MEAN) {
                    out[0] /= a.length;
                }
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached FP64 reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.reduce(primitive, a, device);
            }
        }

        double[] inAd = a.clone();
        double[] out = new double[1];
        try {
            TaskGraph tg = new TaskGraph("jax4j_reduce_" + primitive + "_d_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inAd);
            tg = switch (primitive) {
                case SUM, MEAN -> tg.task("k", TornadoVMBackend::reduceSumD, inAd, out);
                default -> throw new UnsupportedOperationException("No TornadoVM FP64 reduction kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, null, null, null, inAd, null, out);
            boundedOffer(pool, newPlan);

            double[] result = new double[1];
            result[0] = out[0];
            if (primitive == Primitive.MEAN) {
                result[0] /= a.length;
            }
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.reduce(primitive, a, device);
        }
    }

    @Override
    public float[] reduceAxis(Primitive primitive, float[] a, int outerSize, int axisSize, int innerSize, Device device) {
        int outLength = outerSize * innerSize;
        CacheKey key = new CacheKey(primitive, a.length, outerSize, axisSize, innerSize, false, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                cached.plan().execute();
                float[] out = new float[outLength];
                System.arraycopy(cached.outF(), 0, out, 0, outLength);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached axis reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.reduceAxis(primitive, a, outerSize, axisSize, innerSize, device);
            }
        }

        float[] inA = a.clone();
        float[] out = new float[outLength];
        try {
            TaskGraph tg = new TaskGraph("jax4j_reduce_axis_" + primitive + "_" + a.length + "_" + outerSize + "_" + innerSize)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inA);
            tg = switch (primitive) {
                case SUM -> tg.task("k", TornadoVMBackend::reduceAxisSum, inA, out, axisSize, innerSize);
                case MEAN -> tg.task("k", TornadoVMBackend::reduceAxisMean, inA, out, axisSize, innerSize);
                default -> throw new UnsupportedOperationException("No TornadoVM reduction kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, inA, null, out, null, null, null);
            boundedOffer(pool, newPlan);

            float[] result = new float[outLength];
            System.arraycopy(out, 0, result, 0, outLength);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM axis reduction compilation failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.reduceAxis(primitive, a, outerSize, axisSize, innerSize, device);
        }
    }

    public static void vectorAdd(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] + b[i];
    }

    public static void vectorSub(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] - b[i];
    }

    public static void vectorMul(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] * b[i];
    }

    public static void vectorDiv(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] / b[i];
    }

    public static void vectorMax(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.max(a[i], b[i]);
    }

    public static void vectorMin(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.min(a[i], b[i]);
    }

    public static void vectorExp(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.exp(a[i]);
    }

    public static void vectorLog(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.log(a[i]);
    }

    public static void vectorSin(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.sin(a[i]);
    }

    public static void vectorCos(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.cos(a[i]);
    }

    public static void vectorTanh(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.tanh(a[i]);
    }

    public static void vectorRelu(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.max(0f, a[i]);
    }

    public static void vectorSigmoid(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) (1.0 / (1.0 + Math.exp(-a[i])));
    }

    public static void vectorSqrt(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) Math.sqrt(a[i]);
    }

    public static void vectorRsqrt(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = (float) (1.0 / Math.sqrt(a[i]));
    }

    @Override
    public float[] matmul(float[] a, float[] b, int m, int k, int n, Device device) {
        CacheKey key = new CacheKey(Primitive.DOT, a.length, m, k, n, false, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k_ -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                System.arraycopy(b, 0, cached.inB(), 0, b.length);
                cached.plan().execute();
                float[] out = new float[m * n];
                System.arraycopy(cached.outF(), 0, out, 0, out.length);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached matmul failed on {}, falling back to host: {}", device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.matmul(a, b, m, k, n, device);
            }
        }

        float[] inA = a.clone();
        float[] inB = b.clone();
        float[] out = new float[m * n];
        try {
            TaskGraph tg = new TaskGraph("jax4j_matmul_" + m + "_" + k + "_" + n)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inA, inB)
                .task("k", TornadoVMBackend::matmulKernel, inA, inB, out, m, k, n)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, inA, inB, out, null, null, null);
            boundedOffer(pool, newPlan);

            float[] result = new float[m * n];
            System.arraycopy(out, 0, result, 0, out.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM matmul failed on {}, falling back to host: {}", device, t.getMessage());
            return HostBackend.INSTANCE.matmul(a, b, m, k, n, device);
        }
    }

    public static void matmulKernel(float[] a, float[] b, float[] c, int m, int k, int n) {
        for (@Parallel int i = 0; i < m; i++) {
            for (@Parallel int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
                c[i * n + j] = sum;
            }
        }
    }

    // -------------------------------------------------------------------
    // FLOAT64 kernels
    // -------------------------------------------------------------------
    // Consumer NVIDIA GPUs (Turing / Ampere / Ada, e.g. the RTX 3050 this
    // was developed against) have FP64 throughput on the order of 1/32
    // to 1/64 of FP32, and OpenCL doubles need the cl_khr_fp64 extension
    // (present on the RTX 3050 via CUDA-backed OpenCL, absent on some
    // integrated Intel devices). Correctness first here; the small-array
    // dispatch gate lives one layer up in ConcreteNDArray.

    @Override
    public double[] binary(Primitive primitive, double[] a, double[] b, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, true, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inAd(), 0, a.length);
                System.arraycopy(b, 0, cached.inBd(), 0, b.length);
                cached.plan().execute();
                double[] out = new double[a.length];
                System.arraycopy(cached.outD(), 0, out, 0, a.length);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached execution failed for FP64 {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.binary(primitive, a, b, device);
            }
        }

        double[] inAd = a.clone();
        double[] inBd = b.clone();
        double[] out = new double[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_d_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inAd, inBd);
            tg = switch (primitive) {
                case ADD -> tg.task("k", TornadoVMBackend::vectorAddD, inAd, inBd, out);
                case SUB -> tg.task("k", TornadoVMBackend::vectorSubD, inAd, inBd, out);
                case MUL -> tg.task("k", TornadoVMBackend::vectorMulD, inAd, inBd, out);
                case DIV -> tg.task("k", TornadoVMBackend::vectorDivD, inAd, inBd, out);
                case MAX -> tg.task("k", TornadoVMBackend::vectorMaxD, inAd, inBd, out);
                case MIN -> tg.task("k", TornadoVMBackend::vectorMinD, inAd, inBd, out);
                default -> throw new UnsupportedOperationException("No FP64 TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, null, null, null, inAd, inBd, out);
            boundedOffer(pool, newPlan);

            double[] result = new double[a.length];
            System.arraycopy(out, 0, result, 0, a.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.binary(primitive, a, b, device);
        }
    }

    @Override
    public double[] unary(Primitive primitive, double[] a, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, 0, 0, 0, true, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inAd(), 0, a.length);
                cached.plan().execute();
                double[] out = new double[a.length];
                System.arraycopy(cached.outD(), 0, out, 0, a.length);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached FP64 execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.unary(primitive, a, device);
            }
        }

        double[] inAd = a.clone();
        double[] out = new double[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_d_" + a.length)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inAd);
            tg = switch (primitive) {
                case EXP  -> tg.task("k", TornadoVMBackend::vectorExpD, inAd, out);
                case LOG  -> tg.task("k", TornadoVMBackend::vectorLogD, inAd, out);
                case SIN  -> tg.task("k", TornadoVMBackend::vectorSinD, inAd, out);
                case COS  -> tg.task("k", TornadoVMBackend::vectorCosD, inAd, out);
                case TANH -> tg.task("k", TornadoVMBackend::vectorTanhD, inAd, out);
                case RELU -> tg.task("k", TornadoVMBackend::vectorReluD, inAd, out);
                case SIGMOID -> tg.task("k", TornadoVMBackend::vectorSigmoidD, inAd, out);
                case SQRT -> tg.task("k", TornadoVMBackend::vectorSqrtD, inAd, out);
                case RSQRT -> tg.task("k", TornadoVMBackend::vectorRsqrtD, inAd, out);
                default -> throw new UnsupportedOperationException("No FP64 TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, null, null, null, inAd, null, out);
            boundedOffer(pool, newPlan);

            double[] result = new double[a.length];
            System.arraycopy(out, 0, result, 0, a.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.unary(primitive, a, device);
        }
    }

    @Override
    public double[] matmul(double[] a, double[] b, int m, int k, int n, Device device) {
        CacheKey key = new CacheKey(Primitive.DOT, a.length, m, k, n, true, device);
        java.util.concurrent.ConcurrentLinkedQueue<CachedPlan> pool =
            PLAN_CACHE.computeIfAbsent(key, k_ -> new java.util.concurrent.ConcurrentLinkedQueue<>());

        CachedPlan cached = pool.poll();
        if (cached != null) {
            try {
                System.arraycopy(a, 0, cached.inAd(), 0, a.length);
                System.arraycopy(b, 0, cached.inBd(), 0, b.length);
                cached.plan().execute();
                double[] out = new double[m * n];
                System.arraycopy(cached.outD(), 0, out, 0, out.length);
                boundedOffer(pool, cached);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached FP64 matmul failed on {}, falling back to host: {}", device, t.getMessage());
                try {
                    cached.plan().close();
                } catch (Throwable ignore) {}
                return HostBackend.INSTANCE.matmul(a, b, m, k, n, device);
            }
        }

        double[] inAd = a.clone();
        double[] inBd = b.clone();
        double[] out = new double[m * n];
        try {
            TaskGraph tg = new TaskGraph("jax4j_matmul_d_" + m + "_" + k + "_" + n)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, inAd, inBd)
                .task("k", TornadoVMBackend::matmulKernelD, inAd, inBd, out, m, k, n)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            TornadoExecutionPlan plan = configuredPlan(itg, device);
            plan.execute();

            CachedPlan newPlan = new CachedPlan(plan, null, null, null, inAd, inBd, out);
            boundedOffer(pool, newPlan);

            double[] result = new double[m * n];
            System.arraycopy(out, 0, result, 0, out.length);
            return result;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 matmul failed on {}, falling back to host: {}", device, t.getMessage());
            return HostBackend.INSTANCE.matmul(a, b, m, k, n, device);
        }
    }

    public static void vectorAddD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] + b[i];
    }

    public static void vectorSubD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] - b[i];
    }

    public static void vectorMulD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] * b[i];
    }

    public static void vectorDivD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = a[i] / b[i];
    }

    public static void vectorMaxD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.max(a[i], b[i]);
    }

    public static void vectorMinD(double[] a, double[] b, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.min(a[i], b[i]);
    }

    public static void vectorExpD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.exp(a[i]);
    }

    public static void vectorLogD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.log(a[i]);
    }

    public static void vectorSinD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.sin(a[i]);
    }

    public static void vectorCosD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.cos(a[i]);
    }

    public static void vectorTanhD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.tanh(a[i]);
    }

    public static void vectorReluD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.max(0.0, a[i]);
    }

    public static void vectorSigmoidD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = 1.0 / (1.0 + Math.exp(-a[i]));
    }

    public static void vectorSqrtD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = Math.sqrt(a[i]);
    }

    public static void vectorRsqrtD(double[] a, double[] c) {
        for (@Parallel int i = 0; i < a.length; i++) c[i] = 1.0 / Math.sqrt(a[i]);
    }

    public static void matmulKernelD(double[] a, double[] b, double[] c, int m, int k, int n) {
        for (@Parallel int i = 0; i < m; i++) {
            for (@Parallel int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
                c[i * n + j] = sum;
            }
        }
    }

    public static void reduceSum(float[] input, @Reduce float[] result) {
        for (@Parallel int i = 0; i < input.length; i++) {
            result[0] += input[i];
        }
    }

    public static void reduceSumD(double[] input, @Reduce double[] result) {
        for (@Parallel int i = 0; i < input.length; i++) {
            result[0] += input[i];
        }
    }

    public static void vectorCopy(float[] a, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) {
            c[i] = a[i];
        }
    }

    public static void reduceAxisSum(float[] input, float[] output, int axisSize, int innerSize) {
        for (@Parallel int o = 0; o < output.length / innerSize; o++) {
            for (@Parallel int inr = 0; inr < innerSize; inr++) {
                float sum = 0f;
                for (int a = 0; a < axisSize; a++) {
                    sum += input[o * axisSize * innerSize + a * innerSize + inr];
                }
                output[o * innerSize + inr] = sum;
            }
        }
    }

    public static void reduceAxisMean(float[] input, float[] output, int axisSize, int innerSize) {
        for (@Parallel int o = 0; o < output.length / innerSize; o++) {
            for (@Parallel int inr = 0; inr < innerSize; inr++) {
                float sum = 0f;
                for (int a = 0; a < axisSize; a++) {
                    sum += input[o * axisSize * innerSize + a * innerSize + inr];
                }
                output[o * innerSize + inr] = sum / axisSize;
            }
        }
    }
}
