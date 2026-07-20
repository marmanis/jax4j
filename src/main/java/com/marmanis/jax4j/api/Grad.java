package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.CheckpointMeta;
import com.marmanis.jax4j.ir.CondMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.PmapMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.ScanMeta;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.ir.WhileMeta;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Logic for reverse-mode automatic differentiation.
 */
public class Grad {

    /**
     * Transforms a function into its gradient function.
     */
    public static Function<NDArray, NDArray> grad(Function<NDArray, NDArray> fn) {
        return (arg) -> {
            Jaxpr jaxpr = JAX.make_jaxpr(fn, arg);
            return backward(jaxpr, List.of(arg)).get(0);
        };
    }

    /**
     * Transforms a two-argument function into a function returning the gradient
     * with respect to the argument selected by {@code argnum} (0 or 1), mirroring
     * JAX's {@code jax.grad(fn, argnums=...)}.
     */
    public static BiFunction<NDArray, NDArray, NDArray> grad(BiFunction<NDArray, NDArray, NDArray> fn, int argnum) {
        if (argnum != 0 && argnum != 1) {
            throw new IllegalArgumentException("argnum must be 0 or 1, got " + argnum);
        }
        return (arg0, arg1) -> {
            Jaxpr jaxpr = JAX.make_jaxpr(fn, arg0, arg1);
            return backward(jaxpr, List.of(arg0, arg1)).get(argnum);
        };
    }

    /**
     * Like {@link #grad(BiFunction, int)} but returns the gradients with respect
     * to both arguments at once.
     */
    public static BiFunction<NDArray, NDArray, NDArray[]> gradBoth(BiFunction<NDArray, NDArray, NDArray> fn) {
        return (arg0, arg1) -> {
            Jaxpr jaxpr = JAX.make_jaxpr(fn, arg0, arg1);
            List<NDArray> grads = backward(jaxpr, List.of(arg0, arg1));
            return new NDArray[]{grads.get(0), grads.get(1)};
        };
    }

    /**
     * Transforms a function over a {@link PyTree} of parameters (e.g.
     * {@code {"w": ..., "b": ...}}) into a function returning the gradient
     * with respect to every leaf, packaged back into the same tree structure.
     * Mirrors {@code jax.grad} applied to a pytree of parameters.
     */
    public static Function<PyTree, PyTree> gradTree(Function<PyTree, NDArray> fn) {
        return (paramsTree) -> {
            List<NDArray> leaves = PyTrees.flatten(paramsTree);

            Tracer.start();
            Jaxpr jaxpr;
            try {
                List<Var> inVars = new ArrayList<>();
                List<NDArray> tracedLeaves = new ArrayList<>();
                for (NDArray leaf : leaves) {
                    Var v = Tracer.current().nextVar(leaf.shape(), leaf.dtype());
                    inVars.add(v);
                    tracedLeaves.add(new TracedNDArray(v));
                }
                PyTree tracedTree = PyTrees.unflatten(paramsTree, tracedLeaves);
                NDArray result = fn.apply(tracedTree);
                Var outVar = ((TracedNDArray) result).getVar();
                jaxpr = Tracer.stop(inVars, List.of(outVar));
            } catch (RuntimeException | Error e) {
                Tracer.abort();
                throw e;
            }

            List<NDArray> grads = backward(jaxpr, leaves);
            return PyTrees.unflatten(paramsTree, grads);
        };
    }

    /**
     * Returns both the function value and its gradient in a single forward+backward pass.
     * Mirrors {@code jax.value_and_grad}: avoids re-tracing and reuses the forward
     * activations computed during {@link #fillForwardValues} for the backward sweep.
     */
    public static Function<NDArray, NDArray[]> value_and_grad(Function<NDArray, NDArray> fn) {
        return (arg) -> {
            Jaxpr jaxpr = JAX.make_jaxpr(fn, arg);
            Map<Integer, NDArray> fwdValues = fillForwardValues(jaxpr, List.of(arg));
            NDArray value = fwdValues.get(jaxpr.outVars().get(0).id());
            NDArray seed = ones(value.shape(), floatDtypeOrDefault(value.dtype()));
            List<NDArray> grads = doBackward(jaxpr, fwdValues, List.of(seed));
            return new NDArray[]{value, grads.get(0)};
        };
    }

    private static List<NDArray> backward(Jaxpr jaxpr, List<NDArray> argValues) {
        Var outVar = jaxpr.outVars().get(0);
        NDArray seed = ones(outVar.shape(), floatDtypeOrDefault(outVar.dtype()));
        return backwardInterpret(jaxpr, argValues, List.of(seed));
    }

    /**
     * Forward-interprets a (possibly multi-input/output) Jaxpr with concrete
     * values, returning its {@code outVars}' values in order. Used both as
     * the forward half of {@link #backwardInterpret} and to actually execute
     * a {@code COND}/{@code WHILE} branch/body sub-Jaxpr once its predicate
     * has been resolved to a concrete value. Also the execution engine
     * behind {@link JAX#jit}'s cached trace hits.
     */
    public static List<NDArray> forwardInterpret(Jaxpr jaxpr, List<NDArray> argValues) {
        Map<Integer, NDArray> values = fillForwardValues(jaxpr, argValues);
        return jaxpr.outVars().stream().map(v -> values.get(v.id())).toList();
    }

    /**
     * Backward-interprets a Jaxpr seeded with one gradient per {@code
     * outVar} (in order), returning one gradient per {@code inVar} (in
     * order, zero-filled for any input that didn't affect the seeded
     * outputs). This generalizes the original single-output {@code
     * grad}/{@code gradTree} backward pass to the multi-input/output Jaxprs
     * that {@code COND}/{@code WHILE} branch and body functions are traced
     * into. Also the execution engine behind {@link JAX#jitGrad}'s cached
     * gradient trace.
     */
    public static List<NDArray> backwardInterpret(Jaxpr jaxpr, List<NDArray> argValues, List<NDArray> seedGrads) {
        return doBackward(jaxpr, fillForwardValues(jaxpr, argValues), seedGrads);
    }

    private static List<NDArray> doBackward(Jaxpr jaxpr, Map<Integer, NDArray> values, List<NDArray> seedGrads) {
        Map<Integer, NDArray> grads = new HashMap<>();
        for (int i = 0; i < jaxpr.outVars().size(); i++) {
            grads.merge(jaxpr.outVars().get(i).id(), seedGrads.get(i), NDArray::add);
        }

        List<Equation> eqs = jaxpr.equations();
        for (int i = eqs.size() - 1; i >= 0; i--) {
            Equation eq = eqs.get(i);
            NDArray[] inputs = eq.inputs().stream().map(v -> values.get(v.id())).toArray(NDArray[]::new);

            if (eq.primitive() == Primitive.SCAN) {
                NDArray gCarryOut = grads.get(eq.outputs().get(0).id());
                NDArray gYsOut = grads.get(eq.outputs().get(1).id());
                if (gCarryOut == null && gYsOut == null) continue;
                List<NDArray> gIns = scanBackward(eq, inputs, gCarryOut, gYsOut);
                grads.merge(eq.inputs().get(0).id(), gIns.get(0), NDArray::add);
                grads.merge(eq.inputs().get(1).id(), gIns.get(1), NDArray::add);
                continue;
            }

            NDArray gOut = grads.get(eq.outputs().get(0).id());
            if (gOut == null) continue;

            List<NDArray> gIns = computeVJPs(eq.primitive(), eq, gOut, inputs);
            for (int j = 0; j < eq.inputs().size(); j++) {
                grads.merge(eq.inputs().get(j).id(), gIns.get(j), NDArray::add);
            }
        }

        List<NDArray> result = new ArrayList<>();
        for (Var inVar : jaxpr.inVars()) {
            result.add(grads.getOrDefault(inVar.id(), zeros(inVar.shape(), floatDtypeOrDefault(inVar.dtype()))));
        }
        return result;
    }

    /** The forward pass shared by {@link #forwardInterpret} and {@link #backwardInterpret}. */
    private static Map<Integer, NDArray> fillForwardValues(Jaxpr jaxpr, List<NDArray> argValues) {
        Map<Integer, NDArray> values = new HashMap<>();
        values.putAll(jaxpr.consts());
        for (int i = 0; i < jaxpr.inVars().size(); i++) {
            values.put(jaxpr.inVars().get(i).id(), argValues.get(i));
        }

        for (Equation eq : jaxpr.equations()) {
            NDArray[] inputs = eq.inputs().stream().map(v -> values.get(v.id())).toArray(NDArray[]::new);
            if (eq.primitive() == Primitive.SCAN) {
                List<NDArray> outs = scanForward(eq, inputs);
                values.put(eq.outputs().get(0).id(), outs.get(0));
                values.put(eq.outputs().get(1).id(), outs.get(1));
            } else {
                NDArray out = executePrimitive(eq.primitive(), eq, inputs);
                values.put(eq.outputs().get(0).id(), out);
            }
        }
        return values;
    }

    /** Runs a {@code SCAN} equation's step function once per leading-axis slice of {@code xs}. */
    private static List<NDArray> scanForward(Equation eq, NDArray[] inputs) {
        ScanMeta m = (ScanMeta) eq.metadata();
        NDArray carry = inputs[0];
        NDArray xs = inputs[1];
        int steps = xs.shape().dimensions()[0];
        Shape xStepShape = ScanUtil.dropLeadingDim(xs.shape());

        List<NDArray> ys = new ArrayList<>(steps);
        for (int t = 0; t < steps; t++) {
            NDArray xt = ScanUtil.sliceLeading(xs, t, xStepShape);
            List<NDArray> stepOut = forwardInterpret(m.stepFn(), List.of(carry, xt));
            carry = stepOut.get(0);
            ys.add(stepOut.get(1));
        }
        return List.of(carry, ScanUtil.stackLeading(ys));
    }

    /**
     * Reverse-mode through a {@code SCAN}: re-runs the forward pass once
     * more to record every step's {@code (carryIn, x)}, then walks the
     * steps in reverse, backward-interpreting the step function at each one
     * with the accumulated carry gradient and that step's slice of {@code
     * gYsOut} as the two seed gradients (mirrors how JAX differentiates
     * {@code lax.scan} via an internal reverse scan).
     */
    private static List<NDArray> scanBackward(Equation eq, NDArray[] inputs, NDArray gCarryFinal, NDArray gYsStacked) {
        ScanMeta m = (ScanMeta) eq.metadata();
        NDArray initCarry = inputs[0];
        NDArray xs = inputs[1];
        int steps = xs.shape().dimensions()[0];
        Shape xStepShape = ScanUtil.dropLeadingDim(xs.shape());
        Shape yStepShape = m.stepFn().outVars().get(1).shape();

        NDArray[] carryIn = new NDArray[steps];
        NDArray[] xIn = new NDArray[steps];
        NDArray carry = initCarry;
        for (int t = 0; t < steps; t++) {
            carryIn[t] = carry;
            xIn[t] = ScanUtil.sliceLeading(xs, t, xStepShape);
            carry = forwardInterpret(m.stepFn(), List.of(carry, xIn[t])).get(0);
        }

        NDArray gCarry = gCarryFinal != null ? gCarryFinal : zeros(initCarry.shape());
        NDArray[] gXs = new NDArray[steps];
        for (int t = steps - 1; t >= 0; t--) {
            NDArray gY = gYsStacked != null ? ScanUtil.sliceLeading(gYsStacked, t, yStepShape) : zeros(yStepShape);
            List<NDArray> gStepIns = backwardInterpret(m.stepFn(), List.of(carryIn[t], xIn[t]), List.of(gCarry, gY));
            gCarry = gStepIns.get(0);
            gXs[t] = gStepIns.get(1);
        }

        return List.of(gCarry, ScanUtil.stackLeading(java.util.Arrays.asList(gXs)));
    }

    private static NDArray executePrimitive(Primitive p, Equation eq, NDArray[] inputs) {
        return switch (p) {
            case ADD -> inputs[0].add(inputs[1]);
            case SUB -> inputs[0].sub(inputs[1]);
            case MUL -> inputs[0].mul(inputs[1]);
            case DIV -> inputs[0].div(inputs[1]);
            case DOT -> inputs[0].dot(inputs[1]);
            case MEAN -> inputs[0].mean();
            case SUM -> inputs[0].sum();
            case SUM_AXIS, MEAN_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                yield p == Primitive.SUM_AXIS
                    ? inputs[0].sum(m.axis(), m.keepDims())
                    : inputs[0].mean(m.axis(), m.keepDims());
            }
            case EXP -> inputs[0].exp();
            case LOG -> inputs[0].log();
            case SIN -> inputs[0].sin();
            case COS -> inputs[0].cos();
            case TANH -> inputs[0].tanh();
            case RELU -> inputs[0].relu();
            case SIGMOID -> inputs[0].sigmoid();
            case COND -> {
                CondMeta m = (CondMeta) eq.metadata();
                Jaxpr branch = (inputs[0].toFloatArray()[0] != 0f) ? m.trueBranch() : m.falseBranch();
                yield forwardInterpret(branch, List.of(inputs[1])).get(0);
            }
            case WHILE -> {
                WhileMeta m = (WhileMeta) eq.metadata();
                NDArray state = inputs[0];
                while (forwardInterpret(m.condFn(), List.of(state)).get(0).toFloatArray()[0] != 0f) {
                    state = forwardInterpret(m.bodyFn(), List.of(state)).get(0);
                }
                yield state;
            }
            case FFI_CALL -> FFI.call((String) eq.metadata(), eq.outputs().get(0).shape(), inputs);
            case PMAP -> {
                PmapMeta m = (PmapMeta) eq.metadata();
                NDArray[] shards = Pmap.split(inputs[0], m.numDevices());
                yield Pmap.runForwardJaxpr(m.bodyJaxpr(), m.devices(), shards);
            }
            case PSUM -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) yield inputs[0];
                yield new ConcreteNDArray(ctx.collective.psum(ctx.deviceIndex, inputs[0].toFloatArray()), inputs[0].shape());
            }
            case ALL_GATHER -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) yield inputs[0];
                yield ctx.collective.allGather(ctx.deviceIndex, inputs[0]);
            }
            case GT -> inputs[0].gt(inputs[1]);
            case GE -> inputs[0].ge(inputs[1]);
            case LT -> inputs[0].lt(inputs[1]);
            case LE -> inputs[0].le(inputs[1]);
            case EQ -> inputs[0].eq(inputs[1]);
            case NE -> inputs[0].ne(inputs[1]);
            case MAX -> inputs[0].max(inputs[1]);
            case MIN -> inputs[0].min(inputs[1]);
            case ARGMAX, ARGMIN -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                yield p == Primitive.ARGMAX ? inputs[0].argmax(m.axis()) : inputs[0].argmin(m.axis());
            }
            case CAST -> inputs[0].astype((DType) eq.metadata());
            case GATHER -> Numpy.takeEager(inputs[0], inputs[1]);
            case CHECKPOINT -> {
                CheckpointMeta m = (CheckpointMeta) eq.metadata();
                yield forwardInterpret(m.subJaxpr(), List.of(inputs[0])).get(0);
            }
            default -> throw new UnsupportedOperationException(p.toString());
        };
    }

    private static List<NDArray> computeVJPs(Primitive p, Equation eq, NDArray gOut, NDArray[] inputs) {
        return switch (p) {
            case ADD -> List.of(broadcastLike(gOut, inputs[0]), broadcastLike(gOut, inputs[1]));
            case SUB -> List.of(broadcastLike(gOut, inputs[0]), broadcastLike(gOut.mul(minusOne(gOut.shape(), gOut.dtype())), inputs[1]));
            case MUL -> List.of(broadcastLike(gOut.mul(inputs[1]), inputs[0]), broadcastLike(gOut.mul(inputs[0]), inputs[1]));
            case DIV -> {
                // d/da (a/b) = 1/b ; d/db (a/b) = -a/b^2
                NDArray a = inputs[0];
                NDArray b = inputs[1];
                NDArray gA = broadcastLike(gOut.div(b), a);
                NDArray gB = broadcastLike(gOut.mul(a).div(b).div(b).mul(minusOne(gOut.shape(), gOut.dtype())), b);
                yield List.of(gA, gB);
            }
            case MEAN -> {
                double n = inputs[0].shape().size();
                NDArray gIn = broadcastLike(gOut.div(scalar(n, inputs[0].dtype())), inputs[0]);
                yield List.of(gIn);
            }
            case SUM -> List.of(broadcastLike(gOut, inputs[0]));
            case SUM_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                yield List.of(axisBroadcastLike(gOut, inputs[0].shape(), m.axis()));
            }
            case MEAN_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                double axisSize = inputs[0].shape().dimensions()[m.axis()];
                NDArray scaled = gOut.div(scalar(axisSize, gOut.dtype()));
                yield List.of(axisBroadcastLike(scaled, inputs[0].shape(), m.axis()));
            }
            case EXP -> List.of(gOut.mul(inputs[0].exp()));
            case LOG -> List.of(gOut.div(inputs[0]));
            case SIN -> List.of(gOut.mul(inputs[0].cos()));
            case COS -> List.of(gOut.mul(inputs[0].sin()).mul(minusOne(gOut.shape(), gOut.dtype())));
            case TANH -> {
                // d/dx tanh(x) = 1 - tanh(x)^2
                NDArray t = inputs[0].tanh();
                yield List.of(gOut.mul(ones(t.shape(), t.dtype()).sub(t.mul(t))));
            }
            case RELU -> List.of(reluGrad(gOut, inputs[0]));
            case SIGMOID -> {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                NDArray s = inputs[0].sigmoid();
                yield List.of(gOut.mul(s.mul(ones(s.shape(), s.dtype()).sub(s))));
            }
            case DOT -> {
                // C = A . B  =>  dA = gOut . B^T ; dB = A^T . gOut
                NDArray a = inputs[0];
                NDArray b = inputs[1];
                yield List.of(gOut.dot(transpose(b)), transpose(a).dot(gOut));
            }
            case COND -> {
                // pred (inputs[0]) is not differentiable, mirroring jax.lax.cond.
                CondMeta m = (CondMeta) eq.metadata();
                Jaxpr branch = (inputs[0].toFloatArray()[0] != 0f) ? m.trueBranch() : m.falseBranch();
                NDArray gOperand = backwardInterpret(branch, List.of(inputs[1]), List.of(gOut)).get(0);
                yield List.of(zerosLike(inputs[0]), gOperand);
            }
            case WHILE -> throw new UnsupportedOperationException(
                "Lax.whileLoop is not reverse-mode differentiable (mirrors jax.lax.while_loop, whose iteration "
                    + "count is itself data-dependent); use Lax.scan or Lax.foriLoop for a differentiable loop.");
            case FFI_CALL -> {
                FFI.FFITarget target = FFI.getTarget((String) eq.metadata());
                if (target == null || target.vjp() == null) {
                    throw new UnsupportedOperationException("No VJP registered for FFI target: " + eq.metadata());
                }
                NDArray[] gIns = target.vjp().apply(gOut, inputs);
                yield List.of(gIns);
            }
            case GT, GE, LT, LE, EQ, NE ->
                // Standard AD convention: comparisons are treated as locally constant,
                // contributing a zero cotangent rather than erroring (mirrors jax.lax.gt etc).
                List.of(zerosLike(inputs[0]), zerosLike(inputs[1]));
            case MAX -> maxMinGrad(gOut, inputs[0], inputs[1], true);
            case MIN -> maxMinGrad(gOut, inputs[0], inputs[1], false);
            case ARGMAX, ARGMIN ->
                // Index of an extremum has zero gradient w.r.t. the values, same convention as above.
                List.of(zerosLike(inputs[0]));
            case PMAP -> {
                PmapMeta m = (PmapMeta) eq.metadata();
                NDArray[] inputShards = Pmap.split(inputs[0], m.numDevices());
                NDArray[] gOutShards  = Pmap.split(gOut, m.numDevices());
                NDArray[] gInputShards = Pmap.runBackwardJaxpr(m.bodyJaxpr(), m.devices(), inputShards, gOutShards);
                yield List.of(Pmap.stack(gInputShards));
            }
            // VJP of psum (all-reduce sum): psum is a linear map S_j = Σ_i x_i[j], so its
            // transpose is dL/dx_i[j] = Σ_k dL/dS_k[j] = psum(gOut)[j]. In plain English:
            // because x_i contributes to every replica's psum output, gOut must be summed
            // across replicas and sent back to x_i — which is just another psum.
            case PSUM -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) yield List.of(gOut);
                yield List.of(new ConcreteNDArray(
                    ctx.collective.psum(ctx.deviceIndex, gOut.toFloatArray()), gOut.shape()));
            }
            // VJP of all_gather: gOut has shape [D, *shard]; this shard's input contributed
            // only at slice [deviceIndex], so its gradient is that slice.
            case ALL_GATHER -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) yield List.of(gOut);
                yield List.of(Pmap.split(gOut, ctx.numDevices)[ctx.deviceIndex]);
            }
            case CAST -> {
                // A true same-floating-dtype cast (FLOAT32->FLOAT32 or
                // FLOAT64->FLOAT64 — the only "no-op" cases) stays
                // differentiable; every other dtype pair has no continuous
                // derivative (there's nothing meaningful to backprop through a
                // BOOL/INT32/INT64 representation), so it gets a zero cotangent —
                // same convention as comparisons/argmax above.
                DType target = (DType) eq.metadata();
                boolean identity = inputs[0].dtype() == target
                    && (target == DType.FLOAT32 || target == DType.FLOAT64);
                yield List.of(identity ? broadcastLike(gOut, inputs[0]) : zerosLike(inputs[0]));
            }
            case GATHER -> {
                // Embedding backward: scatter-add gOut's rows back into a
                // zero-initialized table-shaped gradient at each looked-up
                // index. Must accumulate (not overwrite) since the same vocab
                // index can repeat within indices. Indices themselves are
                // INT32/non-differentiable, so they get a zero cotangent.
                NDArray table = inputs[0];
                NDArray indices = inputs[1];
                yield List.of(gatherBackward(gOut, table.shape(), indices), zerosLike(indices));
            }
            case CHECKPOINT -> {
                // Rematerialization: re-run the forward pass to get the activations
                // needed for the backward pass, then differentiate through them.
                CheckpointMeta m = (CheckpointMeta) eq.metadata();
                yield List.of(backwardInterpret(m.subJaxpr(), List.of(inputs[0]), List.of(gOut)).get(0));
            }
            default -> throw new UnsupportedOperationException(p.toString());
        };
    }

    /**
     * Scatter-add VJP for {@link Numpy#take}: {@code gOut} has shape
     * {@code [...indices.shape(), dim]}; for every flat position {@code p} in
     * {@code indices}, accumulates {@code gOut}'s row {@code p} into
     * {@code gTable[indices[p], :]}. Accumulation (not assignment) is required
     * since the same table row can be looked up more than once.
     */
    private static NDArray gatherBackward(NDArray gOut, Shape tableShape, NDArray indices) {
        int dim = tableShape.dimensions()[1];
        float[] gTable = new float[(int) tableShape.size()];
        int[] idx = indices.toIntArray();
        float[] g = gOut.toFloatArray();
        for (int p = 0; p < idx.length; p++) {
            int row = idx[p];
            for (int d = 0; d < dim; d++) {
                gTable[row * dim + d] += g[p * dim + d];
            }
        }
        return new ConcreteNDArray(gTable, tableShape);
    }

    /**
     * VJP for {@code max}/{@code min}: the gradient flows entirely to whichever
     * operand "won" at each element (ties go to {@code a}, matching {@code
     * NDArray.max}/{@code min}'s javadoc).
     */
    private static List<NDArray> maxMinGrad(NDArray gOut, NDArray a, NDArray b, boolean isMax) {
        Shape outShape = gOut.shape();
        if (a.dtype() == DType.FLOAT64) {
            double[] g = gOut.toDoubleArray();
            double[] av = a.toDoubleArray();
            double[] bv = b.toDoubleArray();
            double[] gA = new double[g.length];
            double[] gB = new double[g.length];
            for (int i = 0; i < g.length; i++) {
                double va = av[a.shape().broadcastIndex(outShape, i)];
                double vb = bv[b.shape().broadcastIndex(outShape, i)];
                boolean aWins = isMax ? va >= vb : va <= vb;
                if (aWins) gA[i] = g[i]; else gB[i] = g[i];
            }
            return List.of(broadcastLike(new ConcreteNDArray(gA, outShape), a),
                            broadcastLike(new ConcreteNDArray(gB, outShape), b));
        }
        float[] g = gOut.toFloatArray();
        float[] av = a.toFloatArray();
        float[] bv = b.toFloatArray();
        float[] gA = new float[g.length];
        float[] gB = new float[g.length];
        for (int i = 0; i < g.length; i++) {
            float va = av[a.shape().broadcastIndex(outShape, i)];
            float vb = bv[b.shape().broadcastIndex(outShape, i)];
            boolean aWins = isMax ? va >= vb : va <= vb;
            if (aWins) gA[i] = g[i]; else gB[i] = g[i];
        }
        return List.of(broadcastLike(new ConcreteNDArray(gA, outShape), a),
                        broadcastLike(new ConcreteNDArray(gB, outShape), b));
    }

    /**
     * Reconciles a gradient {@code g} with {@code target}'s original shape.
     * If {@code target} is smaller (it was broadcast up during the forward pass),
     * sums the contributions back down. If {@code target} is larger (e.g. the
     * MEAN/SUM vjp seeds a scalar gradient that must spread across every input
     * element), expands {@code g} back out via broadcasting.
     */
    private static NDArray broadcastLike(NDArray g, NDArray target) {
        Shape gShape = g.shape();
        Shape tShape = target.shape();
        if (gShape.equals(tShape)) return g;

        if (g.dtype() == DType.FLOAT64) {
            if (gShape.size() == tShape.size()) {
                return new ConcreteNDArray(g.toDoubleArray(), tShape);
            }
            double[] gData = g.toDoubleArray();
            double[] result = new double[(int) tShape.size()];
            if (gShape.size() < tShape.size()) {
                for (int i = 0; i < result.length; i++) {
                    result[i] = gData[gShape.broadcastIndex(tShape, i)];
                }
            } else {
                for (int i = 0; i < gData.length; i++) {
                    result[tShape.broadcastIndex(gShape, i)] += gData[i];
                }
            }
            return new ConcreteNDArray(result, tShape);
        }

        if (gShape.size() == tShape.size()) {
            // Same total size, different rank/shape: a pure reshape, no reduction needed.
            return new ConcreteNDArray(g.toFloatArray(), tShape);
        }

        float[] gData = g.toFloatArray();
        float[] result = new float[(int) tShape.size()];

        if (gShape.size() < tShape.size()) {
            // g came from a smaller shape; broadcast it back out to fill target.
            for (int i = 0; i < result.length; i++) {
                result[i] = gData[gShape.broadcastIndex(tShape, i)];
            }
        } else {
            // target was broadcast up to gShape; sum contributions back down.
            for (int i = 0; i < gData.length; i++) {
                result[tShape.broadcastIndex(gShape, i)] += gData[i];
            }
        }
        return new ConcreteNDArray(result, tShape);
    }

    /** d/dx relu(x) = 1 if x > 0 else 0; no comparison primitive exists, so this drops to raw arrays. */
    private static NDArray reluGrad(NDArray gOut, NDArray x) {
        if (x.dtype() == DType.FLOAT64) {
            double[] g = gOut.toDoubleArray();
            double[] xv = x.toDoubleArray();
            double[] result = new double[xv.length];
            for (int i = 0; i < xv.length; i++) result[i] = xv[i] > 0 ? g[i] : 0.0;
            return new ConcreteNDArray(result, x.shape());
        }
        float[] g = gOut.toFloatArray();
        float[] xv = x.toFloatArray();
        float[] result = new float[xv.length];
        for (int i = 0; i < xv.length; i++) result[i] = xv[i] > 0 ? g[i] : 0f;
        return new ConcreteNDArray(result, x.shape());
    }

    /**
     * Broadcasts a gradient computed from a {@code SUM_AXIS}/{@code MEAN_AXIS}
     * output back across the reduced {@code axis} of {@code originalShape}.
     * Works regardless of whether the forward op used {@code keepDims} — {@code g}'s
     * flat data always has {@code outerSize * innerSize} elements either way —
     * by replaying the same outer/axis/inner block decomposition as the
     * forward reduction in {@code ConcreteNDArray.reduceAxis}.
     */
    private static NDArray axisBroadcastLike(NDArray g, Shape originalShape, int axis) {
        int[] dims = originalShape.dimensions();
        int axisSize = dims[axis];
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= dims[i];
        int innerSize = 1;
        for (int i = axis + 1; i < dims.length; i++) innerSize *= dims[i];

        if (g.dtype() == DType.FLOAT64) {
            double[] gData = g.toDoubleArray();
            double[] result = new double[(int) originalShape.size()];
            for (int o = 0; o < outerSize; o++) {
                for (int in = 0; in < innerSize; in++) {
                    double v = gData[o * innerSize + in];
                    for (int a = 0; a < axisSize; a++) {
                        result[o * axisSize * innerSize + a * innerSize + in] = v;
                    }
                }
            }
            return new ConcreteNDArray(result, originalShape);
        }

        float[] gData = g.toFloatArray();
        float[] result = new float[(int) originalShape.size()];
        for (int o = 0; o < outerSize; o++) {
            for (int in = 0; in < innerSize; in++) {
                float v = gData[o * innerSize + in];
                for (int a = 0; a < axisSize; a++) {
                    result[o * axisSize * innerSize + a * innerSize + in] = v;
                }
            }
        }
        return new ConcreteNDArray(result, originalShape);
    }

    private static NDArray transpose(NDArray m) {
        int rows = m.shape().dimensions()[0];
        int cols = m.shape().dimensions()[1];
        if (m.dtype() == DType.FLOAT64) {
            double[] in = m.toDoubleArray();
            double[] out = new double[in.length];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    out[j * rows + i] = in[i * cols + j];
                }
            }
            return new ConcreteNDArray(out, new Shape(cols, rows));
        }
        float[] in = m.toFloatArray();
        float[] out = new float[in.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[j * rows + i] = in[i * cols + j];
            }
        }
        return new ConcreteNDArray(out, new Shape(cols, rows));
    }

    public record ForwardAdResult(List<NDArray> primals, List<NDArray> tangents) {}

    public static ForwardAdResult forwardAd(Jaxpr jaxpr, List<NDArray> argValues, List<NDArray> tangentValues) {
        Map<Integer, NDArray> primals = new HashMap<>();
        Map<Integer, NDArray> tangents = new HashMap<>();

        for (Map.Entry<Integer, NDArray> entry : jaxpr.consts().entrySet()) {
            primals.put(entry.getKey(), entry.getValue());
            tangents.put(entry.getKey(), zerosLike(entry.getValue()));
        }

        for (int i = 0; i < jaxpr.inVars().size(); i++) {
            primals.put(jaxpr.inVars().get(i).id(), argValues.get(i));
            tangents.put(jaxpr.inVars().get(i).id(), tangentValues.get(i));
        }

        for (Equation eq : jaxpr.equations()) {
            NDArray[] inPrimals = eq.inputs().stream().map(v -> primals.get(v.id())).toArray(NDArray[]::new);
            NDArray[] inTangents = eq.inputs().stream().map(v -> tangents.get(v.id())).toArray(NDArray[]::new);

            if (eq.primitive() == Primitive.SCAN) {
                List<NDArray> outs = scanJVP(eq, inPrimals, inTangents);
                primals.put(eq.outputs().get(0).id(), outs.get(0));
                primals.put(eq.outputs().get(1).id(), outs.get(1));
                tangents.put(eq.outputs().get(0).id(), outs.get(2));
                tangents.put(eq.outputs().get(1).id(), outs.get(3));
                continue;
            }

            NDArray outPrimal = executePrimitive(eq.primitive(), eq, inPrimals);
            NDArray outTangent = computeJVP(eq.primitive(), eq, outPrimal, inPrimals, inTangents);

            primals.put(eq.outputs().get(0).id(), outPrimal);
            tangents.put(eq.outputs().get(0).id(), outTangent);
        }

        List<NDArray> outPrimals = jaxpr.outVars().stream().map(v -> primals.get(v.id())).toList();
        List<NDArray> outTangents = jaxpr.outVars().stream().map(v -> tangents.get(v.id())).toList();
        return new ForwardAdResult(outPrimals, outTangents);
    }

    private static List<NDArray> scanJVP(Equation eq, NDArray[] inPrimals, NDArray[] inTangents) {
        ScanMeta m = (ScanMeta) eq.metadata();
        NDArray carry = inPrimals[0];
        NDArray xs = inPrimals[1];
        NDArray carryT = inTangents[0];
        NDArray xsT = inTangents[1];

        int steps = xs.shape().dimensions()[0];
        Shape xStepShape = ScanUtil.dropLeadingDim(xs.shape());

        List<NDArray> ysList = new ArrayList<>(steps);
        List<NDArray> ysTList = new ArrayList<>(steps);

        for (int t = 0; t < steps; t++) {
            NDArray xt = ScanUtil.sliceLeading(xs, t, xStepShape);
            NDArray xtT = ScanUtil.sliceLeading(xsT, t, xStepShape);

            ForwardAdResult stepOut = forwardAd(m.stepFn(), List.of(carry, xt), List.of(carryT, xtT));
            carry = stepOut.primals().get(0);
            carryT = stepOut.tangents().get(0);
            ysList.add(stepOut.primals().get(1));
            ysTList.add(stepOut.tangents().get(1));
        }

        return List.of(carry, ScanUtil.stackLeading(ysList), carryT, ScanUtil.stackLeading(ysTList));
    }

    private static NDArray computeJVP(Primitive p, Equation eq, NDArray primalOut, NDArray[] primals, NDArray[] tangents) {
        DType dtype = primalOut.dtype();
        Shape shape = primalOut.shape();
        switch (p) {
            case ADD -> {
                return tangents[0].add(tangents[1]);
            }
            case SUB -> {
                return tangents[0].sub(tangents[1]);
            }
            case MUL -> {
                return tangents[0].mul(primals[1]).add(primals[0].mul(tangents[1]));
            }
            case DIV -> {
                NDArray b = primals[1];
                return tangents[0].mul(b).sub(primals[0].mul(tangents[1])).div(b.mul(b));
            }
            case MEAN -> {
                return tangents[0].mean();
            }
            case SUM -> {
                return tangents[0].sum();
            }
            case SUM_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                return tangents[0].sum(m.axis(), m.keepDims());
            }
            case MEAN_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                return tangents[0].mean(m.axis(), m.keepDims());
            }
            case EXP -> {
                return tangents[0].mul(primalOut);
            }
            case LOG -> {
                return tangents[0].div(primals[0]);
            }
            case SIN -> {
                return tangents[0].mul(primals[0].cos());
            }
            case COS -> {
                return tangents[0].mul(primals[0].sin()).mul(minusOne(shape, dtype));
            }
            case TANH -> {
                NDArray t = primalOut;
                return tangents[0].mul(ones(shape, dtype).sub(t.mul(t)));
            }
            case RELU -> {
                NDArray zero = zeros(primals[0].shape(), primals[0].dtype());
                return tangents[0].mul(primals[0].gt(zero).astype(primals[0].dtype()));
            }
            case SIGMOID -> {
                NDArray s = primalOut;
                return tangents[0].mul(s.mul(ones(shape, dtype).sub(s)));
            }
            case DOT -> {
                return tangents[0].dot(primals[1]).add(primals[0].dot(tangents[1]));
            }
            case COND -> {
                CondMeta m = (CondMeta) eq.metadata();
                Jaxpr branch = (primals[0].toFloatArray()[0] != 0f) ? m.trueBranch() : m.falseBranch();
                ForwardAdResult branchOut = forwardAd(branch, List.of(primals[1]), List.of(tangents[1]));
                return branchOut.tangents().get(0);
            }
            case WHILE -> throw new UnsupportedOperationException("Lax.whileLoop is not forward-mode differentiable");
            case CAST -> {
                DType target = (DType) eq.metadata();
                return tangents[0].astype(target);
            }
            case GATHER -> {
                return Numpy.takeEager(tangents[0], primals[1]);
            }
            case GT, GE, LT, LE, EQ, NE, ARGMAX, ARGMIN -> {
                return zeros(shape, dtype);
            }
            case CHECKPOINT -> {
                CheckpointMeta m = (CheckpointMeta) eq.metadata();
                ForwardAdResult cpOut = forwardAd(m.subJaxpr(), List.of(primals[0]), List.of(tangents[0]));
                return cpOut.tangents().get(0);
            }
            case PMAP -> {
                PmapMeta m = (PmapMeta) eq.metadata();
                NDArray[] primalShards = Pmap.split(primals[0], m.numDevices());
                NDArray[] tangentShards = Pmap.split(tangents[0], m.numDevices());
                NDArray[] outPrimalShards = new NDArray[m.numDevices()];
                NDArray[] outTangentShards = new NDArray[m.numDevices()];
                for (int i = 0; i < m.numDevices(); i++) {
                    ForwardAdResult res = forwardAd(m.bodyJaxpr(), List.of(primalShards[i]), List.of(tangentShards[i]));
                    outPrimalShards[i] = res.primals().get(0);
                    outTangentShards[i] = res.tangents().get(0);
                }
                return Pmap.stack(outTangentShards);
            }
            case PSUM -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) return tangents[0];
                return new ConcreteNDArray(ctx.collective.psum(ctx.deviceIndex, tangents[0].toFloatArray()), tangents[0].shape());
            }
            case ALL_GATHER -> {
                PmapContext ctx = PmapContext.current();
                if (ctx == null) return tangents[0];
                return ctx.collective.allGather(ctx.deviceIndex, tangents[0]);
            }
            default -> throw new UnsupportedOperationException("No JVP rule defined for primitive: " + p);
        }
    }

    /** {@code dtype} if it's a floating dtype, else FLOAT32 — the safe default for an
     * inert zero-gradient placeholder on a non-differentiable (BOOL/INT32/INT64) input. */
    private static DType floatDtypeOrDefault(DType dtype) {
        return (dtype == DType.FLOAT32 || dtype == DType.FLOAT64) ? dtype : DType.FLOAT32;
    }

    private static NDArray scalar(double v, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> new ConcreteNDArray(new float[]{(float) v}, new Shape(1));
            case FLOAT64 -> new ConcreteNDArray(new double[]{v}, new Shape(1));
            default -> throw new IllegalArgumentException("scalar() only supports floating dtypes, got " + dtype);
        };
    }

    private static NDArray ones(Shape shape, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> {
                float[] data = new float[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = 1.0f;
                yield new ConcreteNDArray(data, shape);
            }
            case FLOAT64 -> {
                double[] data = new double[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = 1.0;
                yield new ConcreteNDArray(data, shape);
            }
            default -> throw new IllegalArgumentException("ones() only supports floating dtypes, got " + dtype);
        };
    }

    private static NDArray zeros(Shape shape, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> new ConcreteNDArray(new float[(int) shape.size()], shape);
            case FLOAT64 -> new ConcreteNDArray(new double[(int) shape.size()], shape);
            default -> throw new IllegalArgumentException("zeros() only supports floating dtypes, got " + dtype);
        };
    }

    /** FLOAT32 zero, the historical/default shape — still used by SCAN (always FLOAT32; see {@code ScanUtil}). */
    private static NDArray zeros(Shape shape) {
        return zeros(shape, DType.FLOAT32);
    }

    /** Zero gradient placeholder matching {@code x}'s own dtype when it's floating, else FLOAT32 (inert default). */
    private static NDArray zerosLike(NDArray x) {
        return zeros(x.shape(), floatDtypeOrDefault(x.dtype()));
    }

    private static NDArray minusOne(Shape shape, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> {
                float[] data = new float[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = -1.0f;
                yield new ConcreteNDArray(data, shape);
            }
            case FLOAT64 -> {
                double[] data = new double[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = -1.0;
                yield new ConcreteNDArray(data, shape);
            }
            default -> throw new IllegalArgumentException("minusOne() only supports floating dtypes, got " + dtype);
        };
    }
}
