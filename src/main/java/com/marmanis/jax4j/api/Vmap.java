package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.Lax;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.CheckpointMeta;
import com.marmanis.jax4j.ir.ConcatMeta;
import com.marmanis.jax4j.ir.CustomVjpMeta;
import com.marmanis.jax4j.ir.GridSample2DMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.PadMeta;
import com.marmanis.jax4j.ir.PmapMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.SliceMeta;
import com.marmanis.jax4j.ir.TransposeMeta;
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
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class Vmap {

    public static Function<NDArray, NDArray> vmap(Function<NDArray, NDArray> fn) {
        return vmap(fn, 0, 0);
    }

    /**
     * One-argument vmap with custom input and output axes.
     */
    public static Function<NDArray, NDArray> vmap(Function<NDArray, NDArray> fn, int inAxis, int outAxis) {
        return (batchedArg) -> {
            NDArray norm = inAxis > 0 ? moveAxisToFront(batchedArg, inAxis) : batchedArg;
            boolean batch = inAxis >= 0;
            Shape batchedShape = norm.shape();
            if (batch && batchedShape.rank() == 0) {
                throw new IllegalArgumentException("vmap requires at least a 1-D array with a leading batch dimension");
            }
            int batchSize = batch ? batchedShape.dimensions()[0] : 1;
            Shape exampleShape = batch
                ? new Shape(Arrays.copyOfRange(batchedShape.dimensions(), 1, batchedShape.rank()))
                : batchedShape;

            Tracer.start();
            Var inVar = Tracer.current().nextVar(exampleShape, norm.dtype());
            NDArray result = fn.apply(new TracedNDArray(inVar));
            Var outVar = ((TracedNDArray) result).getVar();
            Jaxpr jaxpr = Tracer.stop(List.of(inVar), List.of(outVar));

            NDArray out = runMulti(jaxpr, List.of(norm), List.of(batch), batchSize).get(0);
            return outAxis > 0 ? moveAxisFromFront(out, outAxis) : out;
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

            NDArray out = runMulti(jaxpr, List.of(norm0, norm1), List.of(batch0, batch1), batchSize).get(0);
            return outAxis > 0 ? moveAxisFromFront(out, outAxis) : out;
        };
    }

    /** Two-argument vmap with {@code in_axes=0} for both arguments and {@code out_axes=0}. */
    public static BiFunction<NDArray, NDArray, NDArray> vmap(BiFunction<NDArray, NDArray, NDArray> fn) {
        return vmap(fn, new int[]{0, 0}, 0);
    }

    /**
     * Tree-vmap transformation: batches leaf values of a PyTree input.
     * {@code inAxes[i]} specifies the batch axis for the i-th leaf in flatten order,
     * and {@code outAxes[j]} specifies the target batch axis for the j-th leaf in the output PyTree.
     */
    public static Function<com.marmanis.jax4j.pytree.PyTree, com.marmanis.jax4j.pytree.PyTree> vmapTree(
            Function<com.marmanis.jax4j.pytree.PyTree, com.marmanis.jax4j.pytree.PyTree> fn,
            int[] inAxes,
            int[] outAxes) {
        return (batchedTree) -> {
            List<NDArray> batchedLeaves = com.marmanis.jax4j.pytree.PyTrees.flatten(batchedTree);
            List<NDArray> normLeaves = new ArrayList<>();
            List<Boolean> isBatched = new ArrayList<>();
            int batchSize = 1;

            for (int i = 0; i < batchedLeaves.size(); i++) {
                NDArray leaf = batchedLeaves.get(i);
                if (leaf == null) {
                    normLeaves.add(null);
                    isBatched.add(false);
                } else {
                    int axis = inAxes[i];
                    if (axis >= 0) {
                        NDArray norm = moveAxisToFront(leaf, axis);
                        normLeaves.add(norm);
                        isBatched.add(true);
                        batchSize = norm.shape().dimensions()[0];
                    } else {
                        normLeaves.add(leaf);
                        isBatched.add(false);
                    }
                }
            }

            Tracer.start();
            List<Var> inVars = new ArrayList<>();
            Jaxpr jaxpr;
            List<NDArray> resultLeavesSample;
            com.marmanis.jax4j.pytree.PyTree resultTree;
            try {
                List<NDArray> tracedLeaves = new ArrayList<>();
                for (int i = 0; i < batchedLeaves.size(); i++) {
                    NDArray leaf = batchedLeaves.get(i);
                    if (leaf == null) {
                        tracedLeaves.add(null);
                    } else {
                        Shape exampleShape = isBatched.get(i)
                            ? new Shape(Arrays.copyOfRange(normLeaves.get(i).shape().dimensions(), 1, normLeaves.get(i).shape().rank()))
                            : leaf.shape();
                        Var v = Tracer.current().nextVar(exampleShape, leaf.dtype());
                        inVars.add(v);
                        tracedLeaves.add(new TracedNDArray(v));
                    }
                }
                com.marmanis.jax4j.pytree.PyTree tracedTree = com.marmanis.jax4j.pytree.PyTrees.unflatten(batchedTree, tracedLeaves);
                resultTree = fn.apply(tracedTree);
                resultLeavesSample = com.marmanis.jax4j.pytree.PyTrees.flatten(resultTree);
                List<Var> outVars = new ArrayList<>();
                for (NDArray gl : resultLeavesSample) {
                    if (gl != null) {
                        outVars.add(((TracedNDArray) gl).getVar());
                    }
                }
                jaxpr = Tracer.stop(inVars, outVars);
            } catch (RuntimeException | Error e) {
                Tracer.abort();
                throw e;
            }

            List<NDArray> nonNullNormLeaves = new ArrayList<>();
            for (NDArray leaf : normLeaves) {
                if (leaf != null) nonNullNormLeaves.add(leaf);
            }
            List<Boolean> nonNullIsBatched = new ArrayList<>();
            for (int i = 0; i < batchedLeaves.size(); i++) {
                if (batchedLeaves.get(i) != null) {
                    nonNullIsBatched.add(isBatched.get(i));
                }
            }

            List<NDArray> rawOutputs = runMulti(jaxpr, nonNullNormLeaves, nonNullIsBatched, batchSize);

            List<NDArray> finalLeaves = new ArrayList<>();
            int rawIdx = 0;
            int outIdx = 0;
            for (NDArray gl : resultLeavesSample) {
                if (gl == null) {
                    finalLeaves.add(null);
                } else {
                    NDArray rawOut = rawOutputs.get(rawIdx++);
                    int targetAxis = outAxes[outIdx++];
                    NDArray oriented = targetAxis > 0 ? moveAxisFromFront(rawOut, targetAxis) : rawOut;
                    finalLeaves.add(oriented);
                }
            }
            return com.marmanis.jax4j.pytree.PyTrees.unflatten(resultTree, finalLeaves);
        };
    }

    /**
     * General multi-input run: initialises the value environment with {@code args}
     * marking each as batched or not, then replays every equation through
     * {@link #applyBatchingRule}. Returns a list of batched outputs corresponding
     * to the outVars.
     */
    private static List<NDArray> runMulti(Jaxpr jaxpr, List<NDArray> args, List<Boolean> isBatched, int batchSize) {
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

        List<NDArray> outputs = new ArrayList<>();
        for (Var outVar : jaxpr.outVars()) {
            NDArray out = values.get(outVar.id());
            if (!batched.contains(outVar.id())) {
                out = broadcastToBatch(out, batchSize);
            }
            outputs.add(out);
        }
        return outputs;
    }

    /**
     * Moves {@code axis} to position 0 by transposing. Supports arbitrary rank.
     */
    static NDArray moveAxisToFront(NDArray a, int axis) {
        if (axis == 0) return a;
        int rank = a.shape().rank();
        int[] axes = new int[rank];
        axes[0] = axis;
        int idx = 1;
        for (int i = 0; i < rank; i++) {
            if (i != axis) {
                axes[idx++] = i;
            }
        }
        return a.transpose(axes);
    }

    /**
     * Moves the front axis (index 0) back to position {@code axis}. Supports arbitrary rank.
     */
    static NDArray moveAxisFromFront(NDArray a, int axis) {
        if (axis == 0) return a;
        int rank = a.shape().rank();
        if (axis >= rank) {
            throw new IllegalArgumentException(
                "outAxis " + axis + " is out of range for output of rank " + rank
                + " (produced by a vmapped function that may reduce rank)");
        }
        int[] axes = new int[rank];
        int idx = 0;
        for (int i = 1; i <= axis; i++) {
            axes[idx++] = i;
        }
        axes[idx++] = 0;
        for (int i = axis + 1; i < rank; i++) {
            axes[idx++] = i;
        }
        return a.transpose(axes);
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
            case GATHER -> {
                if (batched[0]) {
                    int B = inputs[0].shape().dimensions()[0];
                    Shape tableExShape = new Shape(Arrays.copyOfRange(inputs[0].shape().dimensions(), 1, inputs[0].shape().rank()));
                    List<NDArray> shards = new ArrayList<>(B);
                    for (int b = 0; b < B; b++) {
                        NDArray slice = ScanUtil.sliceLeading(inputs[0], b, tableExShape);
                        shards.add(Numpy.takeEager(slice, inputs[1]));
                    }
                    yield ScanUtil.stackLeading(shards);
                } else {
                    yield Numpy.takeEager(inputs[0], inputs[1]);
                }
            }
            case CHECKPOINT -> {
                // Inline the checkpoint body into the batched trace by re-interpreting
                // the sub-Jaxpr through runMulti — equivalent to vmap(fn) where fn is
                // the checkpointed body (the rematerialization hint is a no-op under vmap).
                CheckpointMeta m = (CheckpointMeta) metadata;
                yield runMulti(m.subJaxpr(), List.of(inputs[0]), List.of(batched[0]), batchSize).get(0);
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
            case RESHAPE -> {
                if (metadata instanceof Shape targetExShape) {
                    if (batched[0]) {
                        int b = inputs[0].shape().dimensions()[0];
                        yield inputs[0].reshape(ScanUtil.prependDim(targetExShape, b));
                    } else {
                        yield inputs[0].reshape(targetExShape);
                    }
                } else {
                    if (batched[0]) {
                        int b = inputs[0].shape().dimensions()[0];
                        long innerSize = inputs[0].shape().size() / b;
                        yield inputs[0].reshape(new Shape(b, (int) innerSize));
                    } else {
                        yield inputs[0];
                    }
                }
            }
            case TRANSPOSE -> {
                TransposeMeta m = (TransposeMeta) metadata;
                if (!batched[0]) yield inputs[0].transpose(m.axes());
                // When batched, the leading dim is batch — shift all axes by 1.
                int[] axes = m.axes();
                int[] baxes = new int[axes.length + 1];
                baxes[0] = 0; // batch axis stays at 0
                for (int i = 0; i < axes.length; i++) baxes[i + 1] = axes[i] + 1;
                yield inputs[0].transpose(baxes);
            }
            case CONCAT -> {
                ConcatMeta m = (ConcatMeta) metadata;
                int axis = batched[0] ? m.axis() + 1 : m.axis();
                yield Numpy.concatenateEager(java.util.Arrays.asList(inputs), axis,
                    computeConcatOutputShape(inputs, axis), inputs[0].dtype());
            }
            case PAD -> {
                PadMeta m = (PadMeta) metadata;
                if (!batched[0]) yield inputs[0].pad(m.padding());
                // Shift padding: no padding on batch dim, existing padding on inner dims
                int[][] batchedPad = new int[m.padding().length + 1][2];
                batchedPad[0] = new int[]{0, 0};
                System.arraycopy(m.padding(), 0, batchedPad, 1, m.padding().length);
                yield inputs[0].pad(batchedPad);
            }
            case SCATTER_ADD -> {
                yield Lax.scatterAddEager(inputs[0], inputs[1], inputs[2]);
            }
            case FFT -> {
                // Apply FFT to each batch element (the packed [N,2] representation)
                yield applyPerBatch(inputs[0], batchSize, x -> Grad.fftPackedExec(x, false));
            }
            case IFFT -> {
                yield applyPerBatch(inputs[0], batchSize, x -> Grad.fftPackedExec(x, true));
            }
            case LINALG_SOLVE -> {
                yield Linalg.solve(inputs[0], inputs[1]);
            }
            case LINALG_SVD -> {
                yield Linalg.svd(inputs[0]).sigma();
            }
            case LINALG_EIG -> {
                yield Linalg.eig(inputs[0])[0];
            }
            case CUSTOM_VJP -> {
                CustomVjpMeta m = (CustomVjpMeta) metadata;
                yield m.fn().apply(inputs[0]);
            }
            case SQRT -> inputs[0].sqrt();
            case RSQRT -> inputs[0].rsqrt();
            case MAX_AXIS, MIN_AXIS -> {
                AxisMeta m = (AxisMeta) metadata;
                int axis = batched[0] ? m.axis() + 1 : m.axis();
                yield p == Primitive.MAX_AXIS ? inputs[0].max(axis, m.keepDims()) : inputs[0].min(axis, m.keepDims());
            }
            case SLICE -> {
                SliceMeta m = (SliceMeta) metadata;
                if (!batched[0]) yield inputs[0].slice(m.starts(), m.stops(), m.steps());
                int B = inputs[0].shape().dimensions()[0];
                int[] bStarts = new int[m.starts().length + 1];
                int[] bStops  = new int[m.stops().length + 1];
                int[] bSteps  = new int[m.steps().length + 1];
                bStarts[0] = 0; bStops[0] = B; bSteps[0] = 1;
                System.arraycopy(m.starts(), 0, bStarts, 1, m.starts().length);
                System.arraycopy(m.stops(),  0, bStops,  1, m.stops().length);
                System.arraycopy(m.steps(),  0, bSteps,  1, m.steps().length);
                yield inputs[0].slice(bStarts, bStops, bSteps);
            }
            case MATMUL -> inputs[0].dot(inputs[1]);
            case GRID_SAMPLE_2D -> {
                GridSample2DMeta m = (GridSample2DMeta) metadata;
                // Both input and grid are already rank-4 tensors carrying a
                // batch dim; vmap prepends another leading axis. Slice per
                // outer index and stitch results back together.
                boolean bIn = batched[0];
                boolean bGr = batched[1];
                NDArray in = inputs[0];
                NDArray gr = inputs[1];
                if (!bIn && !bGr) {
                    yield GridSampling.gridSample2dForward(in, gr, m.paddingMode());
                }
                int D = bIn ? in.shape().dimensions()[0] : gr.shape().dimensions()[0];
                Shape inEx = bIn ? new Shape(Arrays.copyOfRange(in.shape().dimensions(), 1, in.shape().rank())) : in.shape();
                Shape grEx = bGr ? new Shape(Arrays.copyOfRange(gr.shape().dimensions(), 1, gr.shape().rank())) : gr.shape();
                List<NDArray> results = new ArrayList<>(D);
                for (int d = 0; d < D; d++) {
                    NDArray inD = bIn ? ScanUtil.sliceLeading(in, d, inEx) : in;
                    NDArray grD = bGr ? ScanUtil.sliceLeading(gr, d, grEx) : gr;
                    results.add(GridSampling.gridSample2dForward(inD, grD, m.paddingMode()));
                }
                yield ScanUtil.stackLeading(results);
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
        if (Tracer.current() != null) {
            NDArray res = batchedArray;
            int rank = res.shape().rank();
            for (int axis = rank - 1; axis >= 1; axis--) {
                res = mean ? res.mean(axis, false) : res.sum(axis, false);
            }
            return res;
        }
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

        if (Tracer.current() != null) {
            List<NDArray> results = new ArrayList<>(batchSize);
            NDArray aReshaped = aBatched ? a.reshape(new Shape(batchSize, M * K)) : null;
            NDArray bReshaped = bBatched ? b.reshape(new Shape(batchSize, K * N)) : null;

            for (int batch = 0; batch < batchSize; batch++) {
                NDArray aSlice;
                if (aBatched) {
                    aSlice = Numpy.take(aReshaped, new ConcreteNDArray(new int[]{batch}, new Shape(1)))
                                   .reshape(new Shape(M, K));
                } else {
                    aSlice = a;
                }

                NDArray bSlice;
                if (bBatched) {
                    bSlice = Numpy.take(bReshaped, new ConcreteNDArray(new int[]{batch}, new Shape(1)))
                                   .reshape(new Shape(K, N));
                } else {
                    bSlice = b;
                }

                NDArray dotVal = aSlice.dot(bSlice);
                results.add(dotVal.reshape(new Shape(1, M * N)));
            }
            NDArray concatenated = Numpy.concatenate(results, 0);
            return concatenated.reshape(new Shape(batchSize, M, N));
        }

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

    private static Shape computeConcatOutputShape(NDArray[] inputs, int axis) {
        int[] outDims = inputs[0].shape().dimensions().clone();
        for (int i = 1; i < inputs.length; i++) outDims[axis] += inputs[i].shape().dimensions()[axis];
        return new Shape(outDims);
    }

    /** Applies a per-example function to each slice along the leading batch dimension. */
    private static NDArray applyPerBatch(NDArray batched, int batchSize, java.util.function.Function<NDArray, NDArray> fn) {
        Shape batchedShape = batched.shape();
        Shape exShape = new Shape(Arrays.copyOfRange(batchedShape.dimensions(), 1, batchedShape.rank()));
        List<NDArray> results = new java.util.ArrayList<>(batchSize);
        for (int b = 0; b < batchSize; b++) {
            NDArray example = ScanUtil.sliceLeading(batched, b, exShape);
            results.add(fn.apply(example));
        }
        return ScanUtil.stackLeading(results);
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
