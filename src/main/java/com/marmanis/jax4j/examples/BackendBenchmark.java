package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.backend.HostBackend;
import com.marmanis.jax4j.backend.TornadoVMBackend;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Wall-clock comparison of the same FP32 numerical work across every
 * execution target jax4j can dispatch to. Reports one column per target:
 *
 * <ul>
 *   <li><b>CPU-1</b> — a single-thread {@link HostBackend} loop.</li>
 *   <li><b>CPU-N</b> — a parallel-stream host loop (Java's Fork/Join).
 *       This is the path {@link com.marmanis.jax4j.core.ConcreteNDArray}
 *       already takes above its {@code PARALLEL_THRESHOLD}.</li>
 *   <li><b>&lt;device&gt;</b> — one column per TornadoVM device discovered
 *       via {@link Device#getDevices()}, labelled with the device name and
 *       its {@link TornadoDeviceType} (CPU/GPU/...). Typically this is the
 *       NVIDIA GPU via PTX and, if POCL or an Intel/NVIDIA OpenCL runtime
 *       exposes the CPU as an OpenCL device, the CPU via OpenCL too.</li>
 * </ul>
 *
 * <p>Each row is one (op, size) pair. Times are median of 5 timed runs
 * after 2 warm-up runs. First-time-touched TornadoVM kernels pay a JIT
 * cost on their first invocation; warm-ups amortize that.
 *
 * <p>Must be launched with the TornadoVM runtime on the module path (see
 * the {@code tornado} Maven profile — activate via
 * {@code mvn -Ptornado exec:exec ...}). Without that, TornadoVM devices
 * won't be discovered and only the CPU-1 / CPU-N columns will be filled.
 */
public class BackendBenchmark {

    private static final int WARMUPS = 2;
    private static final int RUNS = 5;
    /** Default matmul side lengths tried when {@code BENCH_MATMUL_SIZES} is unset. */
    private static final int[] DEFAULT_MATMUL_SIZES = {256, 512, 1024, 2048, 4096, 8192};
    /**
     * At and above this side length, drop CPU-1 from the matmul comparison.
     * A single-threaded 8192³ = 5.5×10¹¹-fmadd naive matmul takes ~30 min per
     * run on a 4 GHz core; times 7 (warmups + timed runs) that's 3.5 hours
     * just for one cell — not useful, and out of the interesting story
     * (CPU-N vs GPU) anyway.
     */
    private static final int LARGE_MATMUL_CUTOFF = 4096;

    private static int[] readMatmulSizes(String csv) {
        if (csv == null || csv.isBlank()) return DEFAULT_MATMUL_SIZES;
        String[] tok = csv.split(",");
        int[] out = new int[tok.length];
        for (int i = 0; i < tok.length; i++) out[i] = Integer.parseInt(tok[i].trim());
        return out;
    }

    public static void main(String[] args) {
        // Optional CLI: --backend=opencl|ptx to bias the automatic selection.
        for (String a : args) {
            if (a.startsWith("--backend=")) {
                Device.useBackend(a.substring("--backend=".length()));
            }
        }

        // Enumerate targets: pure host baselines + every TornadoVM device.
        // Device.host() is our single-thread baseline vessel (its Device
        // handle is only used to label the row — the host loops don't care
        // which Device is passed in since they run the same Java code).
        List<Target> targets = new ArrayList<>();
        targets.add(new Target("CPU-1", Device.host(), Kind.HOST_SEQUENTIAL));
        targets.add(new Target("CPU-N", Device.host(), Kind.HOST_PARALLEL));
        for (Device d : Device.getDevices()) {
            TornadoDevice td = d.getTornadoDevice();
            String label = td == null ? d.getName() : (td.getDeviceName() + " (" + typeLabel(td.getDeviceType()) + ")");
            targets.add(new Target(label, d, Kind.TORNADOVM));
        }

        System.out.println("BackendBenchmark: FP32 comparison across "
            + targets.size() + " execution target" + (targets.size() == 1 ? "" : "s"));
        System.out.println("Warm-ups: " + WARMUPS + ", timed runs: " + RUNS + " (median reported)");
        System.out.println();

        java.util.Random rng = new java.util.Random(0xC0FFEE);

        // Env-var switches (see the class Javadoc for how to set them):
        //   BENCH_MATMUL_ONLY=1  — skip the elementwise section entirely
        //   BENCH_MATMUL_SIZES=csv — override the matmul size list, e.g. "4096,8192"
        boolean matmulOnly = "1".equals(System.getenv("BENCH_MATMUL_ONLY"));
        int[] matmulSizes = readMatmulSizes(System.getenv("BENCH_MATMUL_SIZES"));

        // Elementwise benchmarks over three sizes. Small (below jax4j's
        // PARALLEL_THRESHOLD): CPU-1 and CPU-N should be identical; PCIe
        // transfer overhead dominates the GPU path. Medium and large:
        // CPU-N and GPU should pull ahead of CPU-1.
        int[] elementwiseSizes = {4_096, 65_536, 1 << 20};
        Primitive[] binaryOps = {Primitive.ADD, Primitive.MUL};
        Primitive[] unaryOps = {Primitive.EXP};

        printHeader(targets);
        if (!matmulOnly) {
            for (int n : elementwiseSizes) {
                float[] a = randomFloats(n, rng);
                float[] b = randomFloats(n, rng);
                for (Primitive p : binaryOps) {
                    runElementwiseBinary(p, a, b, targets);
                }
                for (Primitive p : unaryOps) {
                    runElementwiseUnary(p, a, targets);
                }
            }
        }

        // Matmul benchmarks. TornadoVM's DOT kernel here is a naive triply-
        // nested loop with @Parallel on the outer index — GPU wins once the
        // O(n³) compute amortizes the transfer floor, around 512-1024 on
        // this class of laptop GPU. At n >= LARGE_MATMUL_CUTOFF the naive
        // single-threaded loop takes tens of minutes per run; skip it so
        // the benchmark stays under an hour.
        for (int m : matmulSizes) {
            float[] mA = randomFloats(m * m, rng);
            float[] mB = randomFloats(m * m, rng);
            List<Target> matmulTargets = (m >= LARGE_MATMUL_CUTOFF)
                ? targets.stream().filter(t -> t.kind != Kind.HOST_SEQUENTIAL).toList()
                : targets;
            runMatmul(mA, mB, m, targets, matmulTargets);
        }

        System.out.println();
        System.out.println("Notes:");
        System.out.println("  CPU-1 uses HostBackend loops single-threaded.");
        System.out.println("  CPU-N uses Java parallel streams — the same path");
        System.out.println("    ConcreteNDArray takes above PARALLEL_THRESHOLD (65,536).");
        System.out.println("  TornadoVM devices dispatch each op as a separate task graph;");
        System.out.println("    small ops are transfer-dominated on discrete GPUs.");
    }

    // ---------------------------------------------------------------
    // Bench runners
    // ---------------------------------------------------------------

    private static void runElementwiseBinary(Primitive p, float[] a, float[] b, List<Target> targets) {
        float[] reference = HostBackend.INSTANCE.binary(p, a, b, Device.host());
        long[] medians = new long[targets.size()];
        double[] maxErrs = new double[targets.size()];
        for (int t = 0; t < targets.size(); t++) {
            Target tgt = targets.get(t);
            long[] samples = new long[RUNS];
            float[] out = null;
            for (int w = 0; w < WARMUPS; w++) runBinary(tgt, p, a, b);
            for (int r = 0; r < RUNS; r++) {
                long t0 = System.nanoTime();
                out = runBinary(tgt, p, a, b);
                samples[r] = System.nanoTime() - t0;
            }
            medians[t] = median(samples);
            maxErrs[t] = maxAbsDiff(reference, out);
        }
        printRow(p.toString(), a.length, targets, medians, maxErrs);
    }

    private static void runElementwiseUnary(Primitive p, float[] a, List<Target> targets) {
        float[] reference = HostBackend.INSTANCE.unary(p, a, Device.host());
        long[] medians = new long[targets.size()];
        double[] maxErrs = new double[targets.size()];
        for (int t = 0; t < targets.size(); t++) {
            Target tgt = targets.get(t);
            long[] samples = new long[RUNS];
            float[] out = null;
            for (int w = 0; w < WARMUPS; w++) runUnary(tgt, p, a);
            for (int r = 0; r < RUNS; r++) {
                long t0 = System.nanoTime();
                out = runUnary(tgt, p, a);
                samples[r] = System.nanoTime() - t0;
            }
            medians[t] = median(samples);
            maxErrs[t] = maxAbsDiff(reference, out);
        }
        printRow(p.toString(), a.length, targets, medians, maxErrs);
    }

    private static void runMatmul(float[] mA, float[] mB, int m,
                                  List<Target> allTargets, List<Target> runTargets) {
        // Reference: only compute the host baseline when we're actually going
        // to compare against it, i.e. when CPU-1 is still in the run set.
        // At n >= LARGE_MATMUL_CUTOFF, the single-thread reference takes
        // longer than the whole rest of the row combined.
        boolean referenceCpu1 = runTargets.stream().anyMatch(t -> t.kind == Kind.HOST_SEQUENTIAL);
        float[] reference = referenceCpu1
            ? HostBackend.INSTANCE.matmul(mA, mB, m, m, m, Device.host())
            : null;
        long[] medians = new long[allTargets.size()];
        double[] maxErrs = new double[allTargets.size()];
        boolean[] ran = new boolean[allTargets.size()];
        // Use CPU-N's output as the fallback reference when CPU-1 is skipped.
        // Numerical drift between the two is <1e-4 at these sizes, small
        // enough to still give an informative maxErr for the other backends.
        float[] fallbackReference = null;
        for (int t = 0; t < allTargets.size(); t++) {
            Target tgt = allTargets.get(t);
            if (!runTargets.contains(tgt)) continue;
            long[] samples = new long[RUNS];
            float[] out = null;
            for (int w = 0; w < WARMUPS; w++) runMatmulOne(tgt, mA, mB, m);
            for (int r = 0; r < RUNS; r++) {
                long t0 = System.nanoTime();
                out = runMatmulOne(tgt, mA, mB, m);
                samples[r] = System.nanoTime() - t0;
            }
            medians[t] = median(samples);
            ran[t] = true;
            if (reference == null && tgt.kind == Kind.HOST_PARALLEL) {
                fallbackReference = out;
            }
            float[] ref = (reference != null) ? reference : fallbackReference;
            maxErrs[t] = (ref != null) ? maxAbsDiff(ref, out) : Double.NaN;
        }
        printRow("MATMUL", m, allTargets, medians, maxErrs, ran);
    }

    private static float[] runBinary(Target t, Primitive p, float[] a, float[] b) {
        return switch (t.kind) {
            case HOST_SEQUENTIAL -> HostBackend.INSTANCE.binary(p, a, b, t.device);
            case HOST_PARALLEL   -> parallelBinary(p, a, b);
            case TORNADOVM       -> TornadoVMBackend.INSTANCE.binary(p, a, b, t.device);
        };
    }

    private static float[] runUnary(Target t, Primitive p, float[] a) {
        return switch (t.kind) {
            case HOST_SEQUENTIAL -> HostBackend.INSTANCE.unary(p, a, t.device);
            case HOST_PARALLEL   -> parallelUnary(p, a);
            case TORNADOVM       -> TornadoVMBackend.INSTANCE.unary(p, a, t.device);
        };
    }

    private static float[] runMatmulOne(Target t, float[] mA, float[] mB, int m) {
        return switch (t.kind) {
            case HOST_SEQUENTIAL -> HostBackend.INSTANCE.matmul(mA, mB, m, m, m, t.device);
            case HOST_PARALLEL   -> parallelMatmul(mA, mB, m);
            case TORNADOVM       -> TornadoVMBackend.INSTANCE.matmul(mA, mB, m, m, m, t.device);
        };
    }

    // ---------------------------------------------------------------
    // Host multi-core paths (mirror ConcreteNDArray's PARALLEL_THRESHOLD path)
    // ---------------------------------------------------------------

    private static float[] parallelBinary(Primitive p, float[] a, float[] b) {
        float[] out = new float[a.length];
        IntStream.range(0, a.length).parallel().forEach(i -> {
            out[i] = switch (p) {
                case ADD -> a[i] + b[i];
                case SUB -> a[i] - b[i];
                case MUL -> a[i] * b[i];
                case DIV -> a[i] / b[i];
                default -> throw new IllegalStateException(p.toString());
            };
        });
        return out;
    }

    private static float[] parallelUnary(Primitive p, float[] a) {
        float[] out = new float[a.length];
        IntStream.range(0, a.length).parallel().forEach(i -> {
            out[i] = switch (p) {
                case EXP -> (float) Math.exp(a[i]);
                case LOG -> (float) Math.log(a[i]);
                case SIN -> (float) Math.sin(a[i]);
                case COS -> (float) Math.cos(a[i]);
                default -> throw new IllegalStateException(p.toString());
            };
        });
        return out;
    }

    private static float[] parallelMatmul(float[] A, float[] B, int m) {
        float[] C = new float[m * m];
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < m; j++) {
                float s = 0;
                for (int k = 0; k < m; k++) s += A[i * m + k] * B[k * m + j];
                C[i * m + j] = s;
            }
        });
        return C;
    }

    // ---------------------------------------------------------------
    // Reporting
    // ---------------------------------------------------------------

    private static void printHeader(List<Target> targets) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-8s %-10s", "op", "size"));
        for (Target t : targets) sb.append(String.format("  %14s", trimLabel(t.label, 14)));
        sb.append("  maxErr");
        System.out.println(sb);
        System.out.println("-".repeat(sb.length()));
    }

    private static void printRow(String op, int size, List<Target> targets, long[] medians, double[] maxErrs) {
        boolean[] ran = new boolean[targets.size()];
        for (int i = 0; i < ran.length; i++) ran[i] = true;
        printRow(op, size, targets, medians, maxErrs, ran);
    }

    private static void printRow(String op, int size, List<Target> targets,
                                 long[] medians, double[] maxErrs, boolean[] ran) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-8s %-10d", op, size));
        for (int i = 0; i < targets.size(); i++) {
            if (ran[i]) {
                sb.append(String.format("  %10.2f ms", medians[i] / 1e6));
            } else {
                sb.append(String.format("  %10s   ", "—"));
            }
        }
        double worstErr = 0;
        for (int i = 0; i < maxErrs.length; i++) {
            if (ran[i] && !Double.isNaN(maxErrs[i])) worstErr = Math.max(worstErr, maxErrs[i]);
        }
        sb.append(String.format("  %.2e", worstErr));
        System.out.println(sb);
    }

    private static String trimLabel(String s, int width) {
        return s.length() <= width ? s : s.substring(0, width - 1) + "…";
    }

    private static String typeLabel(TornadoDeviceType t) {
        return switch (t) {
            case CPU -> "CPU";
            case GPU -> "GPU";
            case FPGA -> "FPGA";
            case ACCELERATOR -> "ACC";
            default -> t.toString();
        };
    }

    // ---------------------------------------------------------------
    // Utilities
    // ---------------------------------------------------------------

    private static float[] randomFloats(int n, java.util.Random rng) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) rng.nextGaussian();
        return v;
    }

    private static long median(long[] samples) {
        long[] sorted = samples.clone();
        java.util.Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static double maxAbsDiff(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return Double.NaN;
        double m = 0;
        for (int i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
        return m;
    }

    private enum Kind { HOST_SEQUENTIAL, HOST_PARALLEL, TORNADOVM }

    private record Target(String label, Device device, Kind kind) {}
}
