package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

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
}
