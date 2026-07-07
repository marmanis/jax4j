package com.marmanis.jax4j.backend;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;

/**
 * Executes a single elementwise primitive over raw float data on a given device.
 * Implementations are looked up per-{@link Device} (see {@code ConcreteNDArray.backendFor}):
 * {@link HostBackend} for {@link Device#host()}, {@link TornadoVMBackend} otherwise.
 */
public interface ExecutionBackend {
    /** Computes {@code a <primitive> b} elementwise; {@code a} and {@code b} must be the same length. */
    float[] binary(Primitive primitive, float[] a, float[] b, Device device);

    /** Computes {@code <primitive>(a)} elementwise. */
    float[] unary(Primitive primitive, float[] a, Device device);

    /** Computes the {@code (m x k) @ (k x n)} matrix product {@code a @ b}, row-major. */
    float[] matmul(float[] a, float[] b, int m, int k, int n, Device device);

    /**
     * FLOAT64 elementwise binary op. Backends that don't want to accelerate
     * FLOAT64 (e.g. because their hardware's FP64 throughput is far lower
     * than FP32, or a kernel isn't written yet) should delegate to
     * {@link HostBackend#INSTANCE}.
     */
    default double[] binary(Primitive primitive, double[] a, double[] b, Device device) {
        return HostBackend.INSTANCE.binary(primitive, a, b, device);
    }

    /** FLOAT64 elementwise unary op. Defaults to host. */
    default double[] unary(Primitive primitive, double[] a, Device device) {
        return HostBackend.INSTANCE.unary(primitive, a, device);
    }

    /**
     * FLOAT64 matmul. Defaults to host. On consumer NVIDIA GPUs the FP64
     * throughput is 1/32–1/64 of FP32, so this kernel is only worth
     * dispatching for very large matrices ({@code m*n >= ~50k}); the
     * caller ({@link com.marmanis.jax4j.core.ConcreteNDArray}) is where
     * that size gate lives.
     */
    default double[] matmul(double[] a, double[] b, int m, int k, int n, Device device) {
        return HostBackend.INSTANCE.matmul(a, b, m, k, n, device);
    }
}
