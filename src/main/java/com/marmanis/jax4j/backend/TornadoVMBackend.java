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
    // Key: (primitive, array-size, isF64).
    // Cached value: the pre-built plan together with its bound float[]
    // buffers so we can copy fresh input data in before re-executing.
    // The TornadoVM transferToDevice(EVERY_EXECUTION, ...) mode reads
    // the array reference it was given at graph-construction time on
    // every execute() call, so writing new values into those arrays and
    // calling execute() again is the correct way to reuse a plan.
    // ----------------------------------------------------------------

    private record CacheKey(Primitive primitive, int size, boolean isF64) {}

    private record CachedPlan(TornadoExecutionPlan plan,
                               float[] inA, float[] inB, float[] outF,
                               double[] inAd, double[] inBd, double[] outD) {}

    private static final java.util.concurrent.ConcurrentHashMap<CacheKey, CachedPlan> PLAN_CACHE =
        new java.util.concurrent.ConcurrentHashMap<>();

    @Override
    public float[] binary(Primitive primitive, float[] a, float[] b, Device device) {
        CacheKey key = new CacheKey(primitive, a.length, false);
        CachedPlan cached = PLAN_CACHE.get(key);
        if (cached != null) {
            // Cache hit: update bound buffers and re-execute.
            // The plan's TaskGraph was built with EVERY_EXECUTION transfer mode,
            // so writing into the same float[] arrays and calling execute() again
            // uploads the fresh data automatically.
            try {
                System.arraycopy(a, 0, cached.inA(), 0, a.length);
                System.arraycopy(b, 0, cached.inB(), 0, b.length);
                cached.plan().execute();
                float[] out = new float[a.length];
                System.arraycopy(cached.outF(), 0, out, 0, a.length);
                return out;
            } catch (Throwable t) {
                log.warn("TornadoVM cached execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
                return HostBackend.INSTANCE.binary(primitive, a, b, device);
            }
        }

        // Cache miss: build, cache, execute.
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
            // Store in cache (don't close the plan — it must stay alive for reuse).
            PLAN_CACHE.putIfAbsent(key, new CachedPlan(plan, inA, inB, out, null, null, null));
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
        float[] out = new float[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a);
            tg = switch (primitive) {
                case EXP -> tg.task("k", TornadoVMBackend::vectorExp, a, out);
                case LOG -> tg.task("k", TornadoVMBackend::vectorLog, a, out);
                case SIN -> tg.task("k", TornadoVMBackend::vectorSin, a, out);
                case COS -> tg.task("k", TornadoVMBackend::vectorCos, a, out);
                case TANH -> tg.task("k", TornadoVMBackend::vectorTanh, a, out);
                case RELU -> tg.task("k", TornadoVMBackend::vectorRelu, a, out);
                case SIGMOID -> tg.task("k", TornadoVMBackend::vectorSigmoid, a, out);
                default -> throw new UnsupportedOperationException("No TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            return out;
        } catch (Throwable t) {
            log.warn("TornadoVM execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.unary(primitive, a, device);
        }
    }

    @Override
    public float[] reduce(Primitive primitive, float[] a, Device device) {
        float[] out = new float[1];
        try {
            TaskGraph tg = new TaskGraph("jax4j_reduce_" + primitive)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a);
            tg = switch (primitive) {
                case SUM, MEAN -> tg.task("k", TornadoVMBackend::reduceSum, a, out);
                default -> throw new UnsupportedOperationException("No TornadoVM reduction kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            if (primitive == Primitive.MEAN) {
                out[0] /= a.length;
            }
            return out;
        } catch (Throwable t) {
            log.warn("TornadoVM reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.reduce(primitive, a, device);
        }
    }

    @Override
    public double[] reduce(Primitive primitive, double[] a, Device device) {
        double[] out = new double[1];
        try {
            TaskGraph tg = new TaskGraph("jax4j_reduce_" + primitive + "_d")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a);
            tg = switch (primitive) {
                case SUM, MEAN -> tg.task("k", TornadoVMBackend::reduceSumD, a, out);
                default -> throw new UnsupportedOperationException("No TornadoVM FP64 reduction kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            if (primitive == Primitive.MEAN) {
                out[0] /= a.length;
            }
            return out;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 reduction failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.reduce(primitive, a, device);
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

    @Override
    public float[] matmul(float[] a, float[] b, int m, int k, int n, Device device) {
        float[] out = new float[m * n];
        try {
            TaskGraph tg = new TaskGraph("jax4j_matmul")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .task("k", TornadoVMBackend::matmulKernel, a, b, out, m, k, n)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            return out;
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
        double[] out = new double[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_d")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b);
            tg = switch (primitive) {
                case ADD -> tg.task("k", TornadoVMBackend::vectorAddD, a, b, out);
                case SUB -> tg.task("k", TornadoVMBackend::vectorSubD, a, b, out);
                case MUL -> tg.task("k", TornadoVMBackend::vectorMulD, a, b, out);
                case DIV -> tg.task("k", TornadoVMBackend::vectorDivD, a, b, out);
                case MAX -> tg.task("k", TornadoVMBackend::vectorMaxD, a, b, out);
                case MIN -> tg.task("k", TornadoVMBackend::vectorMinD, a, b, out);
                default -> throw new UnsupportedOperationException("No FP64 TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            return out;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.binary(primitive, a, b, device);
        }
    }

    @Override
    public double[] unary(Primitive primitive, double[] a, Device device) {
        double[] out = new double[a.length];
        try {
            TaskGraph tg = new TaskGraph("jax4j_" + primitive + "_d")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a);
            tg = switch (primitive) {
                case EXP  -> tg.task("k", TornadoVMBackend::vectorExpD, a, out);
                case LOG  -> tg.task("k", TornadoVMBackend::vectorLogD, a, out);
                case SIN  -> tg.task("k", TornadoVMBackend::vectorSinD, a, out);
                case COS  -> tg.task("k", TornadoVMBackend::vectorCosD, a, out);
                case TANH -> tg.task("k", TornadoVMBackend::vectorTanhD, a, out);
                case RELU -> tg.task("k", TornadoVMBackend::vectorReluD, a, out);
                case SIGMOID -> tg.task("k", TornadoVMBackend::vectorSigmoidD, a, out);
                default -> throw new UnsupportedOperationException("No FP64 TornadoVM kernel for " + primitive);
            };
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            return out;
        } catch (Throwable t) {
            log.warn("TornadoVM FP64 execution failed for {} on {}, falling back to host: {}", primitive, device, t.getMessage());
            return HostBackend.INSTANCE.unary(primitive, a, device);
        }
    }

    @Override
    public double[] matmul(double[] a, double[] b, int m, int k, int n, Device device) {
        double[] out = new double[m * n];
        try {
            TaskGraph tg = new TaskGraph("jax4j_matmul_d")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
                .task("k", TornadoVMBackend::matmulKernelD, a, b, out, m, k, n)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            ImmutableTaskGraph itg = tg.snapshot();
            try (TornadoExecutionPlan plan = configuredPlan(itg, device)) {
                plan.execute();
            }
            return out;
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
}
