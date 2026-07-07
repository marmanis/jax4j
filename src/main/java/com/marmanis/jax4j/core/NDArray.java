package com.marmanis.jax4j.core;

/**
 * Interface for N-dimensional arrays in jax4j.
 * Supports both concrete data and traced variables for transformations.
 */
public interface NDArray {
    Shape shape();
    DType dtype();
    Device device();

    /**
     * Returns an array with the same data placed on {@code device}, mirroring
     * {@code jax.device_put}. For {@link ConcreteNDArray} this is a cheap re-tag
     * (jax4j always holds host {@code float[]} data; TornadoVM manages on-device
     * buffers only for the duration of a single dispatched execution).
     */
    NDArray to(Device device);

    // Basic Arithmetic
    NDArray add(NDArray other);
    NDArray sub(NDArray other);
    NDArray mul(NDArray other);
    NDArray div(NDArray other);

    // Linear Algebra
    NDArray dot(NDArray other);

    // Math Functions
    NDArray exp();
    NDArray log();
    NDArray sin();
    NDArray cos();

    // Activations
    NDArray tanh();
    NDArray relu();
    NDArray sigmoid();

    // Reductions
    NDArray sum();
    NDArray mean();

    /**
     * Reduces along a single {@code axis} (negative axes count from the end,
     * e.g. {@code -1} for the last dimension), mirroring
     * {@code numpy.sum(axis=, keepdims=)}. If {@code keepDims} the reduced
     * dimension is kept as size 1 (so the result still broadcasts against the
     * original shape); otherwise it's dropped and rank decreases by one.
     */
    NDArray sum(int axis, boolean keepDims);
    NDArray mean(int axis, boolean keepDims);

    /** Equivalent to {@code sum(axis, false)}. */
    default NDArray sum(int axis) { return sum(axis, false); }
    /** Equivalent to {@code mean(axis, false)}. */
    default NDArray mean(int axis) { return mean(axis, false); }

    // Comparisons (elementwise, broadcasting; result is a real DType.BOOL
    // array). Mirroring jax.lax.gt/ge/lt/le/eq/ne, these have no gradient
    // (treated as a constant zero cotangent, the standard AD convention for
    // comparisons — see Grad's VJP rules) rather than erroring if used inside
    // a differentiated function. Both operands must share a dtype (FLOAT32,
    // INT32, FLOAT64, or INT64); mismatches throw — use astype() to convert
    // explicitly.
    NDArray gt(NDArray other);
    NDArray ge(NDArray other);
    NDArray lt(NDArray other);
    NDArray le(NDArray other);
    NDArray eq(NDArray other);
    NDArray ne(NDArray other);

    /**
     * Elementwise maximum/minimum, mirroring {@code jax.lax.max}/{@code
     * jax.lax.min}. Differentiable: the gradient flows entirely to whichever
     * operand "won" at each element (ties go to the first argument).
     */
    NDArray max(NDArray other);
    NDArray min(NDArray other);

    /** Equivalent to {@code this.max(lo).min(hi)}, mirroring {@code jax.numpy.clip}. */
    default NDArray clip(NDArray lo, NDArray hi) { return this.max(lo).min(hi); }

    /**
     * Index of the maximum/minimum element along {@code axis} (negative axes
     * count from the end), mirroring {@code jax.numpy.argmax}/{@code argmin}.
     * The reduced dimension is always dropped (no {@code keepDims}). Indices
     * are returned as a real {@code DType.INT32} array (regardless of the
     * input's floating width) and, like the comparisons above, have no
     * gradient. Input must be FLOAT32 or FLOAT64.
     */
    NDArray argmax(int axis);
    NDArray argmin(int axis);

    /**
     * Returns the underlying data as a flat array. Caution: this might trigger
     * data transfer from device to host. Throws {@code IllegalStateException}
     * if {@link #dtype()} is not {@link DType#FLOAT32} — use {@link #astype}
     * to convert first.
     */
    float[] toFloatArray();

    /** Equivalent to {@link #toFloatArray()} for {@link DType#INT32} arrays. */
    int[] toIntArray();

    /** Equivalent to {@link #toFloatArray()} for {@link DType#BOOL} arrays. */
    boolean[] toBoolArray();

    /** Equivalent to {@link #toFloatArray()} for {@link DType#FLOAT64} arrays. */
    double[] toDoubleArray();

    /** Equivalent to {@link #toFloatArray()} for {@link DType#INT64} arrays. */
    long[] toLongArray();

    /**
     * Converts to a different dtype, mirroring {@code jax.numpy.astype}. This
     * is the principled way to move between FLOAT32/INT32/BOOL/FLOAT64/INT64 —
     * jax4j does not silently promote or reinterpret across dtypes anywhere
     * else (e.g. arithmetic ops require both operands to share the same
     * floating dtype and throw otherwise), so casts are always explicit and
     * visible at the call site.
     *
     * <p>Casting to a narrower numeric type truncates toward zero (FLOAT→INT,
     * matching numpy/jax {@code .astype(intN)}) or narrows bit width (INT64→
     * INT32, which can overflow — caller's responsibility, same as Java's own
     * narrowing cast); any numeric type → BOOL is "nonzero"; BOOL → any
     * numeric type is 1/0; casting to the same dtype is the identity (and,
     * uniquely among casts, stays differentiable for the two floating dtypes —
     * see {@code Grad}'s {@code CAST} VJP rule).
     */
    NDArray astype(DType target);
}
