package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.ConcatMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.ArrayList;
import java.util.List;

/**
 * NumPy-style helpers mirroring {@code jax.numpy}. {@link #where} is composed
 * from existing differentiable primitives ({@code MUL}/{@code ADD}/{@code SUB})
 * rather than needing its own primitive/VJP rule — {@code cond} is cast to
 * FLOAT32 internally if it's the (typical) {@code DType.BOOL} result of a
 * comparison, then used as a masked blend of {@code x} and {@code y}.
 *
 * <p>{@link #take} is a genuine new primitive ({@code GATHER}) — an embedding
 * lookup table can't be built from existing elementwise ops, since it needs an
 * actual indexed read.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Numpy {
    private Numpy() {}

    private static boolean isTracing() {
        return Tracer.current() != null;
    }

    private static Var toVar(NDArray a) {
        if (a instanceof TracedNDArray t) return t.getVar();
        return Tracer.current().nextConstant(a);
    }

    /** Equivalent to {@code cond ? x : y} elementwise, mirroring {@code jax.numpy.where}. */
    public static NDArray where(NDArray cond, NDArray x, NDArray y) {
        NDArray condFloat = cond.dtype() == DType.BOOL ? cond.astype(DType.FLOAT32) : cond;
        NDArray one = new ConcreteNDArray(new float[]{1f}, new Shape(1));
        NDArray invCond = one.sub(condFloat);
        return condFloat.mul(x).add(invCond.mul(y));
    }

    /**
     * Embedding-table lookup, mirroring {@code jax.numpy.take(table, indices,
     * axis=0)}: {@code table} is {@code [vocab, dim]} (FLOAT32), {@code
     * indices} is INT32 of any shape (commonly {@code [batch, seqLen]} token
     * ids), and the result is {@code [...indices.shape(), dim]} (FLOAT32).
     * Differentiable w.r.t. {@code table} via a scatter-add VJP (see {@code
     * Grad}'s {@code GATHER} case) — the standard embedding backward.
     */
    public static NDArray take(NDArray table, NDArray indices) {
        if (isTracing()) {
            Tracer tracer = Tracer.current();
            Var tableVar = toVar(table);
            Var indicesVar = toVar(indices);
            Shape outShape = appendDim(indices.shape(), table.shape().dimensions()[1]);
            Var outVar = tracer.nextVar(outShape, table.dtype());
            tracer.addEquation(new Equation(List.of(tableVar, indicesVar), List.of(outVar), Primitive.GATHER, null));
            return new TracedNDArray(outVar);
        }
        return takeEager(table, indices);
    }

    /** The eager gather loop, shared with {@code Grad}'s forward re-interpretation and {@code Vmap}'s batching rule. */
    static NDArray takeEager(NDArray table, NDArray indices) {
        int dim = table.shape().dimensions()[1];
        int[] idx = indices.toIntArray();
        float[] tableData = table.toFloatArray();
        float[] out = new float[idx.length * dim];
        for (int p = 0; p < idx.length; p++) {
            System.arraycopy(tableData, idx[p] * dim, out, p * dim, dim);
        }
        return new ConcreteNDArray(out, appendDim(indices.shape(), dim));
    }

    private static Shape appendDim(Shape shape, int n) {
        int[] dims = shape.dimensions();
        int[] out = new int[dims.length + 1];
        System.arraycopy(dims, 0, out, 0, dims.length);
        out[dims.length] = n;
        return new Shape(out);
    }

    /**
     * Concatenates a list of arrays along {@code axis}, mirroring
     * {@code jax.numpy.concatenate}. All arrays must share the same dtype
     * and the same shape in every dimension except {@code axis}.
     * Differentiable: the VJP splits the incoming gradient along {@code axis}.
     */
    public static NDArray concatenate(List<NDArray> arrays, int axis) {
        if (arrays.isEmpty()) throw new IllegalArgumentException("concatenate: empty list");
        DType dtype = arrays.get(0).dtype();
        int rank = arrays.get(0).shape().rank();
        int normAxis = axis < 0 ? rank + axis : axis;

        int[] outDims = arrays.get(0).shape().dimensions().clone();
        for (int i = 1; i < arrays.size(); i++) {
            outDims[normAxis] += arrays.get(i).shape().dimensions()[normAxis];
        }
        Shape outShape = new Shape(outDims);

        if (isTracing()) {
            Tracer tracer = Tracer.current();
            List<Var> inVars = arrays.stream().map(a -> {
                if (a instanceof TracedNDArray t) return t.getVar();
                return tracer.nextConstant(a);
            }).toList();
            Var outVar = tracer.nextVar(outShape, dtype);
            tracer.addEquation(new Equation(inVars, List.of(outVar), Primitive.CONCAT, new ConcatMeta(normAxis)));
            return new TracedNDArray(outVar);
        }

        return concatenateEager(arrays, normAxis, outShape, dtype);
    }

    /** Eager concatenation; also used by Grad's forward re-interpretation. */
    public static NDArray concatenateEager(List<NDArray> arrays, int axis, Shape outShape, DType dtype) {
        int size = (int) outShape.size();
        int rank = outShape.rank();
        int[] outDims = outShape.dimensions();

        // Compute output strides (row-major)
        int[] outStrides = new int[rank];
        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];

        if (dtype == DType.FLOAT64) {
            double[] out = new double[size];
            int offset = 0;
            for (NDArray arr : arrays) {
                double[] src = arr.toDoubleArray();
                int[] srcDims = arr.shape().dimensions();
                int[] srcStrides = new int[rank];
                srcStrides[rank - 1] = 1;
                for (int i = rank - 2; i >= 0; i--) srcStrides[i] = srcStrides[i + 1] * srcDims[i + 1];
                int srcSize = (int) arr.shape().size();
                for (int srcFlat = 0; srcFlat < srcSize; srcFlat++) {
                    int rem = srcFlat;
                    int[] coords = new int[rank];
                    for (int i = rank - 1; i >= 0; i--) { coords[i] = rem % srcDims[i]; rem /= srcDims[i]; }
                    coords[axis] += offset;
                    int outFlat = 0;
                    for (int i = 0; i < rank; i++) outFlat += coords[i] * outStrides[i];
                    out[outFlat] = src[srcFlat];
                }
                offset += srcDims[axis];
            }
            return new ConcreteNDArray(out, outShape);
        }

        float[] out = new float[size];
        int offset = 0;
        for (NDArray arr : arrays) {
            float[] src = arr.toFloatArray();
            int[] srcDims = arr.shape().dimensions();
            int srcSize = (int) arr.shape().size();
            int[] srcStrides = new int[rank];
            srcStrides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--) srcStrides[i] = srcStrides[i + 1] * srcDims[i + 1];
            for (int srcFlat = 0; srcFlat < srcSize; srcFlat++) {
                int rem = srcFlat;
                int[] coords = new int[rank];
                for (int i = rank - 1; i >= 0; i--) { coords[i] = rem % srcDims[i]; rem /= srcDims[i]; }
                coords[axis] += offset;
                int outFlat = 0;
                for (int i = 0; i < rank; i++) outFlat += coords[i] * outStrides[i];
                out[outFlat] = src[srcFlat];
            }
            offset += srcDims[axis];
        }
        return new ConcreteNDArray(out, outShape);
    }

    /**
     * Stacks arrays along a new axis, mirroring {@code jax.numpy.stack}.
     * Each input must have the same shape; a new axis of size
     * {@code arrays.size()} is inserted at position {@code axis}.
     */
    public static NDArray stack(List<NDArray> arrays, int axis) {
        if (arrays.isEmpty()) throw new IllegalArgumentException("stack: empty list");
        List<NDArray> expanded = new ArrayList<>(arrays.size());
        for (NDArray a : arrays) expanded.add(a.reshape(insertDim(a.shape(), axis)));
        return concatenate(expanded, axis);
    }

    private static Shape insertDim(Shape shape, int axis) {
        int rank = shape.rank();
        int normAxis = axis < 0 ? rank + 1 + axis : axis;
        int[] dims = shape.dimensions();
        int[] out = new int[rank + 1];
        for (int i = 0; i < normAxis; i++) out[i] = dims[i];
        out[normAxis] = 1;
        for (int i = normAxis; i < rank; i++) out[i + 1] = dims[i];
        return new Shape(out);
    }

    /**
     * Slices {@code x} along {@code axis} in range {@code [start, end)}.
     * Used internally by Grad's CONCAT VJP.
     */
    static NDArray sliceAxis(NDArray x, int axis, int start, int end) {
        int rank = x.shape().rank();
        int[] dims = x.shape().dimensions();
        int[] outDims = dims.clone();
        outDims[axis] = end - start;
        Shape outShape = new Shape(outDims);
        int size = (int) outShape.size();
        int[] outStrides = new int[rank];
        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];
        int[] inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * dims[i + 1];

        if (x.dtype() == DType.FLOAT64) {
            double[] in = x.toDoubleArray();
            double[] out = new double[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int rem = outFlat;
                int inFlat = 0;
                for (int i = rank - 1; i >= 0; i--) {
                    int coord = rem % outDims[i];
                    rem /= outDims[i];
                    int inCoord = (i == axis) ? coord + start : coord;
                    inFlat += inCoord * inStrides[i];
                }
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape);
        }
        float[] in = x.toFloatArray();
        float[] out = new float[size];
        for (int outFlat = 0; outFlat < size; outFlat++) {
            int rem = outFlat;
            int inFlat = 0;
            for (int i = rank - 1; i >= 0; i--) {
                int coord = rem % outDims[i];
                rem /= outDims[i];
                int inCoord = (i == axis) ? coord + start : coord;
                inFlat += inCoord * inStrides[i];
            }
            out[outFlat] = in[inFlat];
        }
        return new ConcreteNDArray(out, outShape);
    }
}
