package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.backend.HostBackend;
import com.marmanis.jax4j.backend.TornadoVMBackend;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;

import java.util.List;

/**
 * Verifies that the FLOAT64 kernels added to {@link TornadoVMBackend} run
 * on the GPU and give correct results (bit-identical to the host loop for
 * add/mul/div; within {@code 1e-12} for transcendentals where the GPU may
 * differ in the last few bits from JDK's {@code StrictMath}-style routines).
 *
 * <p>Must be launched via the {@code tornado} script, since the TornadoVM
 * runtime hooks aren't loaded by a plain {@code java} invocation. Example:
 * <pre>
 *   tornado --classpath ... com.marmanis.jax4j.examples.Fp64GpuVerify
 * </pre>
 */
public class Fp64GpuVerify {
    public static void main(String[] args) {
        List<Device> devices = Device.getDevices();
        if (devices.isEmpty()) {
            System.out.println("No TornadoVM devices discovered; nothing to verify.");
            return;
        }
        Device gpu = devices.get(0);
        System.out.println("Device: " + gpu.getName());
        System.out.println();

        int n = 1 << 20; // ~1M doubles
        double[] a = new double[n];
        double[] b = new double[n];
        java.util.Random rng = new java.util.Random(1);
        for (int i = 0; i < n; i++) {
            a[i] = rng.nextGaussian();
            b[i] = rng.nextGaussian() * 0.5 + 1.0;
        }

        checkBinary(Primitive.ADD, a, b, gpu, 0.0);
        checkBinary(Primitive.MUL, a, b, gpu, 0.0);
        checkBinary(Primitive.SUB, a, b, gpu, 0.0);
        checkBinary(Primitive.DIV, a, b, gpu, 1e-14);
        checkUnary(Primitive.EXP, a, gpu, 1e-12);
        checkUnary(Primitive.SIN, a, gpu, 1e-12);
        checkUnary(Primitive.COS, a, gpu, 1e-12);
        checkUnary(Primitive.TANH, a, gpu, 1e-10);

        System.out.println();
        // A small matmul sanity check.
        int m = 128, k = 128;
        double[] mA = new double[m * k];
        double[] mB = new double[k * m];
        for (int i = 0; i < mA.length; i++) mA[i] = rng.nextGaussian();
        for (int i = 0; i < mB.length; i++) mB[i] = rng.nextGaussian();
        long t0 = System.nanoTime();
        double[] hostC = HostBackend.INSTANCE.matmul(mA, mB, m, k, m, gpu);
        long t1 = System.nanoTime();
        double[] gpuC = TornadoVMBackend.INSTANCE.matmul(mA, mB, m, k, m, gpu);
        long t2 = System.nanoTime();
        double maxErr = 0;
        for (int i = 0; i < hostC.length; i++) maxErr = Math.max(maxErr, Math.abs(hostC[i] - gpuC[i]));
        System.out.printf("matmul 128x128 FP64: host %.2f ms, gpu %.2f ms, maxErr %.2e%n",
            (t1 - t0) / 1e6, (t2 - t1) / 1e6, maxErr);
    }

    private static void checkBinary(Primitive p, double[] a, double[] b, Device gpu, double tol) {
        long t0 = System.nanoTime();
        double[] hostOut = HostBackend.INSTANCE.binary(p, a, b, gpu);
        long t1 = System.nanoTime();
        double[] gpuOut = TornadoVMBackend.INSTANCE.binary(p, a, b, gpu);
        long t2 = System.nanoTime();
        double maxErr = 0;
        for (int i = 0; i < a.length; i++) maxErr = Math.max(maxErr, Math.abs(hostOut[i] - gpuOut[i]));
        System.out.printf("%-6s FP64 (n=%d): host %.2f ms, gpu %.2f ms, maxErr %.2e %s%n",
            p, a.length, (t1 - t0) / 1e6, (t2 - t1) / 1e6, maxErr,
            maxErr <= tol ? "OK" : "FAIL(tol=" + tol + ")");
    }

    private static void checkUnary(Primitive p, double[] a, Device gpu, double tol) {
        long t0 = System.nanoTime();
        double[] hostOut = HostBackend.INSTANCE.unary(p, a, gpu);
        long t1 = System.nanoTime();
        double[] gpuOut = TornadoVMBackend.INSTANCE.unary(p, a, gpu);
        long t2 = System.nanoTime();
        double maxErr = 0;
        for (int i = 0; i < a.length; i++) maxErr = Math.max(maxErr, Math.abs(hostOut[i] - gpuOut[i]));
        System.out.printf("%-6s FP64 (n=%d): host %.2f ms, gpu %.2f ms, maxErr %.2e %s%n",
            p, a.length, (t1 - t0) / 1e6, (t2 - t1) / 1e6, maxErr,
            maxErr <= tol ? "OK" : "FAIL(tol=" + tol + ")");
    }
}
