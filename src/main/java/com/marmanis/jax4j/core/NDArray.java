package com.marmanis.jax4j.core;

/**
 * Interface for N-dimensional arrays in jax4j.
 * Supports both concrete data and traced variables for transformations.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
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

    /** Elementwise {@code sqrt(x)}, mirroring {@code jax.numpy.sqrt}. Differentiable
     * (VJP: {@code 0.5 / sqrt(x)}). Input must be non-negative floating-point. */
    NDArray sqrt();

    /** Elementwise reciprocal square root {@code 1/sqrt(x)}, mirroring
     * {@code jax.lax.rsqrt}. Differentiable and numerically stable — prefer over
     * {@code x.sqrt().reciprocal()} or {@code exp(-0.5*log(x))}. */
    NDArray rsqrt();

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

    /**
     * Reduces along a single {@code axis} using max/min, mirroring
     * {@code numpy.max(axis=, keepdims=)} / {@code numpy.min(...)}.
     * Differentiable: the gradient flows to the argmax/argmin position(s) within
     * each reduction slice; ties split the gradient equally.
     */
    NDArray max(int axis, boolean keepDims);
    NDArray min(int axis, boolean keepDims);
    /** Equivalent to {@code max(axis, false)}. */
    default NDArray max(int axis) { return max(axis, false); }
    /** Equivalent to {@code min(axis, false)}. */
    default NDArray min(int axis) { return min(axis, false); }

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
     * Returns a view with the same data but a different shape, mirroring
     * {@code jax.numpy.reshape}. The total number of elements must be
     * unchanged. Works for all dtypes.
     */
    NDArray reshape(Shape newShape);

    /** Convenience overload: {@code reshape(new Shape(dims))}. */
    default NDArray reshape(int... dims) { return reshape(new Shape(dims)); }

    /**
     * Permutes the axes of this array, mirroring {@code jax.numpy.transpose}.
     * {@code axes[i]} is the axis of the input that maps to output axis {@code i}.
     * For a 2-D matrix, {@code transpose(1, 0)} is the usual matrix transpose.
     */
    NDArray transpose(int... axes);

    /**
     * Reverses all axes (the default numpy/jax transpose with no arguments),
     * i.e. {@code transpose(rank-1, rank-2, ..., 0)}.
     */
    default NDArray transpose() {
        int rank = shape().rank();
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) axes[i] = rank - 1 - i;
        return transpose(axes);
    }

    /**
     * Pads every dimension with zeros. {@code padding[i] = {before_i, after_i}}
     * gives the number of zero-elements to prepend and append to dimension {@code i}.
     * Mirroring {@code jax.numpy.pad} with {@code mode='constant'} and
     * {@code constant_values=0}.
     */
    NDArray pad(int[][] padding);

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

    /**
     * General slicing (numpy-style) with per-axis {@code starts}, {@code stops}, and
     * positive {@code steps}. All three arrays must have length equal to the input rank.
     * {@code stops[i] == -1} is a sentinel meaning "to end of axis". Differentiable:
     * the VJP zero-pads the gradient back to the original input shape (injecting zeros
     * between kept positions for step &gt; 1).
     */
    NDArray slice(int[] starts, int[] stops, int[] steps);

    /** Equivalent to {@code slice(starts, stops, ones)}. */
    default NDArray slice(int[] starts, int[] stops) {
        int[] steps = new int[starts.length];
        java.util.Arrays.fill(steps, 1);
        return slice(starts, stops, steps);
    }

    /** Slice one axis and keep every other axis full. */
    default NDArray sliceAxis(int axis, int start, int stop) {
        int rank = shape().rank();
        int norm = axis < 0 ? rank + axis : axis;
        int[] starts = new int[rank];
        int[] stops = new int[rank];
        int[] steps = new int[rank];
        int[] dims = shape().dimensions();
        for (int i = 0; i < rank; i++) {
            starts[i] = 0;
            stops[i] = dims[i];
            steps[i] = 1;
        }
        starts[norm] = start;
        stops[norm] = stop;
        return slice(starts, stops, steps);
    }

    /**
     * Pick one index along {@code axis}, reducing the rank of the result by one.
     * Equivalent to {@code sliceAxis(axis, index, index+1)} followed by a squeeze
     * along that axis.
     */
    default NDArray sliceIndex(int axis, int index) {
        int rank = shape().rank();
        int norm = axis < 0 ? rank + axis : axis;
        NDArray s = sliceAxis(norm, index, index + 1);
        int[] dims = s.shape().dimensions();
        int[] out = new int[dims.length - 1];
        for (int i = 0, j = 0; i < dims.length; i++) {
            if (i != norm) out[j++] = dims[i];
        }
        return s.reshape(new Shape(out));
    }

    /**
     * Matrix multiply. For rank &le; 2 both operands, equivalent to {@link #dot}.
     * For rank &gt; 2, treats the last two axes as matrix axes and broadcasts all
     * leading axes numpy-style: {@code A[..., M, K] @ B[..., K, N] -> [broadcast(...), M, N]}.
     * Differentiable in both operands.
     */
    default NDArray matmul(NDArray other) { return dot(other); }
}
