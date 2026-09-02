package com.marmanis.jax4j.backend;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;

/**
 * Executes a single elementwise primitive over raw float data on a given device.
 * Implementations are looked up per-{@link Device} (see {@code ConcreteNDArray.backendFor}):
 * {@link HostBackend} for {@link Device#host()}, {@link TornadoVMBackend} otherwise.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface ExecutionBackend {

    /**
     * Pick the appropriate backend for a given device: {@link TornadoVMBackend}
     * when the device is TornadoVM-backed, otherwise {@link HostBackend}. This
     * is the same rule {@code ConcreteNDArray} uses internally; exposed here
     * so higher-level composites (convolutions, custom ops) can dispatch to
     * the right kernel without depending on {@code ConcreteNDArray}'s
     * package-private internals.
     */
    static ExecutionBackend forDevice(com.marmanis.jax4j.core.Device device) {
        return device.getTornadoDevice() != null ? TornadoVMBackend.INSTANCE : HostBackend.INSTANCE;
    }

    /** Computes {@code a <primitive> b} elementwise; {@code a} and {@code b} must be the same length. */
    float[] binary(Primitive primitive, float[] a, float[] b, Device device);

    /** Computes {@code <primitive>(a)} elementwise. */
    float[] unary(Primitive primitive, float[] a, Device device);

    /** Computes the {@code (m x k) @ (k x n)} matrix product {@code a @ b}, row-major. */
    float[] matmul(float[] a, float[] b, int m, int k, int n, Device device);

    /** Computes the reduction of array `a` to a scalar. */
    float[] reduce(Primitive primitive, float[] a, Device device);

    /** Computes the axis reduction of array `a` along `axis` with given outer, axis, and inner sizes. */
    float[] reduceAxis(Primitive primitive, float[] a, int outerSize, int axisSize, int innerSize, Device device);

    /**
     * FLOAT64 reduction. Defaults to host.
     */
    default double[] reduce(Primitive primitive, double[] a, Device device) {
        return HostBackend.INSTANCE.reduce(primitive, a, device);
    }

    /** FLOAT64 axis reduction. Defaults to host. */
    default double[] reduceAxis(Primitive primitive, double[] a, int outerSize, int axisSize, int innerSize, Device device) {
        return HostBackend.INSTANCE.reduceAxis(primitive, a, outerSize, axisSize, innerSize, device);
    }

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
