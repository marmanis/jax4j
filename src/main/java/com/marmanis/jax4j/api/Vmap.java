package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.CheckpointMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.PmapMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;

import java.util.ArrayList;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Automatic batching, mirroring the role of {@code jax.vmap}: traces {@code fn}
 * once against a per-example (unbatched) placeholder, then replays the traced
 * Jaxpr against a real batched input where every value carries a leading batch
 * dimension (or doesn't, if it came from the closure rather than the mapped
 * argument). Each primitive gets its own "batching rule", just as each
 * primitive has its own VJP rule in {@link Grad}.
 *
 * <p>Unlike {@code grad}, this implementation only supports {@code in_axes=0,
 * out_axes=0} (the most common case) and executes eagerly: it does not compose
 * with further tracing (e.g. {@code grad(vmap(f))}).
 */
public class Vmap {

    public static Function<NDArray, NDArray> vmap(Function<NDArray, NDArray> fn) {
        return (batchedArg) -> {
            Shape batchedShape = batchedArg.shape();
            if (batchedShape.rank() == 0) {
                throw new IllegalArgumentException("vmap requires at least a 1-D array with a leading batch dimension");
            }
            int batchSize = batchedShape.dimensions()[0];
            Shape exampleShape = new Shape(Arrays.copyOfRange(batchedShape.dimensions(), 1, batchedShape.rank()));

            Tracer.start();
            Var inVar = Tracer.current().nextVar(exampleShape, batchedArg.dtype());
            NDArray result = fn.apply(new TracedNDArray(inVar));
            Var outVar = ((TracedNDArray) result).getVar();
            Jaxpr jaxpr = Tracer.stop(List.of(inVar), List.of(outVar));

            return runMulti(jaxpr, List.of(batchedArg), List.of(true), batchSize);
        };
    }

    /**
     * Two-argument vmap with per-argument axis control.
     *
     * <p>{@code inAxes[i]} is the axis of the i-th argument that carries the
     * batch dimension, or {@code -1} to leave that argument unbatched (the
     * jax4j equivalent of Python JAX's {@code None}).  {@code outAxis} is the
     * axis where the batch dimension should appear in the output (0 = leading,
     * 1 = second axis, etc.).
     *
     * <p>Axis transposition is limited to 2-D arrays; for higher-rank inputs
     * use {@code in_axes=0} and pre-transpose manually.
     */
    public static BiFunction<NDArray, NDArray, NDArray> vmap(
            BiFunction<NDArray, NDArray, NDArray> fn,
            int[] inAxes,
            int outAxis) {
        return (arg0, arg1) -> {
            NDArray norm0 = inAxes[0] > 0 ? moveAxisToFront(arg0, inAxes[0]) : arg0;
            NDArray norm1 = inAxes[1] > 0 ? moveAxisToFront(arg1, inAxes[1]) : arg1;

            boolean batch0 = inAxes[0] >= 0;
            boolean batch1 = inAxes[1] >= 0;
            int batchSize = batch0 ? norm0.shape().dimensions()[0] : norm1.shape().dimensions()[0];

            Shape exShape0 = batch0
                ? new Shape(Arrays.copyOfRange(norm0.shape().dimensions(), 1, norm0.shape().rank()))
                : norm0.shape();
            Shape exShape1 = batch1
                ? new Shape(Arrays.copyOfRange(norm1.shape().dimensions(), 1, norm1.shape().rank()))
                : norm1.shape();

            Tracer.start();
            Var v0 = Tracer.current().nextVar(exShape0, norm0.dtype());
            Var v1 = Tracer.current().nextVar(exShape1, norm1.dtype());
            NDArray result = fn.apply(new TracedNDArray(v0), new TracedNDArray(v1));
            Var outVar = ((TracedNDArray) result).getVar();
            Jaxpr jaxpr = Tracer.stop(List.of(v0, v1), List.of(outVar));

            NDArray out = runMulti(jaxpr, List.of(norm0, norm1), List.of(batch0, batch1), batchSize);
            return outAxis > 0 ? moveAxisToFront(out, outAxis) : out;
        };
    }

    /** Two-argument vmap with {@code in_axes=0} for both arguments and {@code out_axes=0}. */
    public static BiFunction<NDArray, NDArray, NDArray> vmap(BiFunction<NDArray, NDArray, NDArray> fn) {
        return vmap(fn, new int[]{0, 0}, 0);
    }

    /**
     * General multi-input run: initialises the value environment with {@code args}
     * marking each as batched or not, then replays every equation through
     * {@link #applyBatchingRule}.
     */
    private static NDArray runMulti(Jaxpr jaxpr, List<NDArray> args, List<Boolean> isBatched, int batchSize) {
        Map<Integer, NDArray> values = new HashMap<>();
        Set<Integer> batched = new HashSet<>();

        values.putAll(jaxpr.consts()); // closed-over constants: never batched
        for (int i = 0; i < args.size(); i++) {
            int varId = jaxpr.inVars().get(i).id();
            values.put(varId, args.get(i));
            if (isBatched.get(i)) batched.add(varId);
        }

        for (Equation eq : jaxpr.equations()) {
            NDArray[] inputs = eq.inputs().stream().map(v -> values.get(v.id())).toArray(NDArray[]::new);
            boolean[] inputIsBatched = new boolean[inputs.length];
            boolean anyBatched = false;
            for (int i = 0; i < inputs.length; i++) {
                inputIsBatched[i] = batched.contains(eq.inputs().get(i).id());
                anyBatched |= inputIsBatched[i];
            }

            NDArray out = applyBatchingRule(eq.primitive(), eq.metadata(), inputs, inputIsBatched, batchSize);
            values.put(eq.outputs().get(0).id(), out);
            if (anyBatched) batched.add(eq.outputs().get(0).id());
        }

        Var outVar = jaxpr.outVars().get(0);
        NDArray out = values.get(outVar.id());
        if (!batched.contains(outVar.id())) {
            out = broadcastToBatch(out, batchSize);
        }
        return out;
    }

    /**
     * Moves {@code axis} to position 0 by transposing. Currently supports
     * 2-D arrays only; {@code axis=0} is a no-op for any rank.
     */
    static NDArray moveAxisToFront(NDArray a, int axis) {
        if (axis == 0) return a;
        int rank = a.shape().rank();
        if (rank == 2 && axis == 1) {
            int[] dims = a.shape().dimensions();
            int rows = dims[0], cols = dims[1];
            float[] src = a.toFloatArray();
            float[] dst = new float[src.length];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    dst[j * rows + i] = src[i * cols + j];
            return new ConcreteNDArray(dst, new Shape(cols, rows));
        }
        throw new UnsupportedOperationException(
            "moveAxisToFront: only axis=0 (any rank) or axis=1 on 2-D arrays is supported; got axis=" + axis + " rank=" + rank);
    }

    /**
     * Dispatches to each primitive's batching rule. Elementwise primitives need
     * no special handling at all: jax4j's existing NumPy-style broadcasting
     * already treats a missing leading dimension as size 1, so "batched (B,...)
     * op unbatched (...)" broadcasts correctly through the ordinary eager ops.
     * Only the reducing ops (SUM/MEAN) and DOT need batch-aware rules, since
     * their unbatched implementations collapse dimensions that must instead be
     * kept per-example here.
     */
    private static NDArray applyBatchingRule(Primitive p, Object metadata, NDArray[] inputs, boolean[] batched, int batchSize) {
        return switch (p) {
            case ADD -> inputs[0].add(inputs[1]);
            case SUB -> inputs[0].sub(inputs[1]);
            case MUL -> inputs[0].mul(inputs[1]);
            case DIV -> inputs[0].div(inputs[1]);
            case EXP -> inputs[0].exp();
            case LOG -> inputs[0].log();
            case SIN -> inputs[0].sin();
            case COS -> inputs[0].cos();
            case TANH -> inputs[0].tanh();
            case RELU -> inputs[0].relu();
            case SIGMOID -> inputs[0].sigmoid();
            case SUM -> batched[0] ? batchedReduce(inputs[0], batchSize, false) : inputs[0].sum();
            case MEAN -> batched[0] ? batchedReduce(inputs[0], batchSize, true) : inputs[0].mean();
            case SUM_AXIS, MEAN_AXIS -> {
                AxisMeta m = (AxisMeta) metadata;
                // A leading batch dim was inserted at 0, so every other axis shifts by one.
                int axis = batched[0] ? m.axis() + 1 : m.axis();
                yield p == Primitive.SUM_AXIS ? inputs[0].sum(axis, m.keepDims()) : inputs[0].mean(axis, m.keepDims());
            }
            case DOT -> batchedDot(inputs[0], batched[0], inputs[1], batched[1], batchSize);
            case GT -> inputs[0].gt(inputs[1]);
            case GE -> inputs[0].ge(inputs[1]);
            case LT -> inputs[0].lt(inputs[1]);
            case LE -> inputs[0].le(inputs[1]);
            case EQ -> inputs[0].eq(inputs[1]);
            case NE -> inputs[0].ne(inputs[1]);
            case MAX -> inputs[0].max(inputs[1]);
            case MIN -> inputs[0].min(inputs[1]);
            case ARGMAX, ARGMIN -> {
                AxisMeta m = (AxisMeta) metadata;
                int axis = batched[0] ? m.axis() + 1 : m.axis();
                yield p == Primitive.ARGMAX ? inputs[0].argmax(axis) : inputs[0].argmin(axis);
            }
            case CAST -> inputs[0].astype((com.marmanis.jax4j.core.DType) metadata);
            // Generalizes for free: indices' shape can already include a
            // leading batch dim (the eager loop only cares about indices'
            // total element count), so a batched lookup needs no special casing.
            case GATHER -> Numpy.takeEager(inputs[0], inputs[1]);
            case CHECKPOINT -> {
                // Inline the checkpoint body into the batched trace by re-interpreting
                // the sub-Jaxpr through runMulti — equivalent to vmap(fn) where fn is
                // the checkpointed body (the rematerialization hint is a no-op under vmap).
                CheckpointMeta m = (CheckpointMeta) metadata;
                yield runMulti(m.subJaxpr(), List.of(inputs[0]), List.of(batched[0]), batchSize);
            }
            case PMAP -> {
                PmapMeta m = (PmapMeta) metadata;
                if (!batched[0]) {
                    NDArray[] shards = Pmap.split(inputs[0], m.numDevices());
                    yield Pmap.runForwardJaxpr(m.bodyJaxpr(), m.devices(), shards);
                }
                // Batched input [B, D, *rest]: run pmap over each batch element.
                // Note: collectives inside the body remain scoped to D shards per example.
                int B = inputs[0].shape().dimensions()[0];
                Shape perBatch = new Shape(Arrays.copyOfRange(
                    inputs[0].shape().dimensions(), 1, inputs[0].shape().rank()));
                List<NDArray> batchResults = new ArrayList<>(B);
                for (int b = 0; b < B; b++) {
                    NDArray single = ScanUtil.sliceLeading(inputs[0], b, perBatch);
                    NDArray[] shards = Pmap.split(single, m.numDevices());
                    batchResults.add(Pmap.runForwardJaxpr(m.bodyJaxpr(), m.devices(), shards));
                }
                yield ScanUtil.stackLeading(batchResults);
            }
            default -> throw new UnsupportedOperationException(
                "vmap has no batching rule for primitive: " + p);
        };
    }

    /**
     * Per-example sum/mean: reduces every axis except the leading batch axis,
     * producing one scalar per batch element instead of collapsing the whole
     * batched array to a single scalar.
     */
    private static NDArray batchedReduce(NDArray batchedArray, int batchSize, boolean mean) {
        float[] data = batchedArray.toFloatArray();
        int perExample = data.length / batchSize;
        float[] out = new float[batchSize];
        for (int b = 0; b < batchSize; b++) {
            float total = 0;
            for (int i = 0; i < perExample; i++) total += data[b * perExample + i];
            out[b] = mean ? total / perExample : total;
        }
        return new ConcreteNDArray(out, new Shape(batchSize));
    }

    /**
     * Batched matrix multiply: if both operands are batched, multiplies
     * matching pairs (A[b] . B[b]); if only one is batched, broadcasts the
     * unbatched matrix across every batch element.
     */
    private static NDArray batchedDot(NDArray a, boolean aBatched, NDArray b, boolean bBatched, int batchSize) {
        if (!aBatched && !bBatched) return a.dot(b);

        int[] aDims = aBatched ? Arrays.copyOfRange(a.shape().dimensions(), 1, a.shape().rank()) : a.shape().dimensions();
        int[] bDims = bBatched ? Arrays.copyOfRange(b.shape().dimensions(), 1, b.shape().rank()) : b.shape().dimensions();
        int M = aDims[0];
        int K = aDims[1];
        int N = bDims[1];

        float[] aData = a.toFloatArray();
        float[] bData = b.toFloatArray();
        float[] out = new float[batchSize * M * N];

        for (int batch = 0; batch < batchSize; batch++) {
            int aOff = aBatched ? batch * M * K : 0;
            int bOff = bBatched ? batch * K * N : 0;
            int outOff = batch * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += aData[aOff + i * K + k] * bData[bOff + k * N + j];
                    }
                    out[outOff + i * N + j] = sum;
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(batchSize, M, N));
    }

    private static NDArray broadcastToBatch(NDArray value, int batchSize) {
        float[] data = value.toFloatArray();
        float[] out = new float[batchSize * data.length];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(data, 0, out, b * data.length, data.length);
        }
        int[] dims = new int[value.shape().rank() + 1];
        dims[0] = batchSize;
        System.arraycopy(value.shape().dimensions(), 0, dims, 1, value.shape().rank());
        return new ConcreteNDArray(out, new Shape(dims));
    }
}
