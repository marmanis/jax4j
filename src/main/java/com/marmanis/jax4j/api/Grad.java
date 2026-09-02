package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.core.PRNGKey;
import java.util.Arrays;
import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.CheckpointMeta;
import com.marmanis.jax4j.ir.ConcatMeta;
import com.marmanis.jax4j.ir.CondMeta;
import com.marmanis.jax4j.ir.Conv2DMeta;
import com.marmanis.jax4j.ir.Conv2DTransposeMeta;
import com.marmanis.jax4j.ir.CustomVjpMeta;
import com.marmanis.jax4j.ir.DepthwiseConv2DMeta;
import com.marmanis.jax4j.ir.GridSample2DMeta;
import com.marmanis.jax4j.ir.Pool2DMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.PadMeta;
import com.marmanis.jax4j.ir.PmapMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.ScanMeta;
import com.marmanis.jax4j.ir.SliceMeta;
import com.marmanis.jax4j.ir.TransposeMeta;
import com.marmanis.jax4j.ir.RandomMeta;
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
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
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
            List<Var> inVars = new ArrayList<>();
            try {
                List<NDArray> tracedLeaves = new ArrayList<>();
                for (NDArray leaf : leaves) {
                    if (leaf == null) {
                        tracedLeaves.add(null);
                    } else {
                        Var v = Tracer.current().nextVar(leaf.shape(), leaf.dtype());
                        inVars.add(v);
                        tracedLeaves.add(new TracedNDArray(v));
                    }
                }
                PyTree tracedTree = PyTrees.unflatten(paramsTree, tracedLeaves);
                NDArray result = fn.apply(tracedTree);
                Var outVar = ((TracedNDArray) result).getVar();
                jaxpr = Tracer.stop(inVars, List.of(outVar));
            } catch (RuntimeException | Error e) {
                Tracer.abort();
                throw e;
            }

            List<NDArray> nonNullLeaves = new ArrayList<>();
            for (NDArray leaf : leaves) {
                if (leaf != null) nonNullLeaves.add(leaf);
            }
            List<NDArray> rawGrads = backward(jaxpr, nonNullLeaves);

            List<NDArray> grads = new ArrayList<>();
            int gradsIdx = 0;
            for (NDArray leaf : leaves) {
                if (leaf == null) {
                    grads.add(null);
                } else {
                    grads.add(rawGrads.get(gradsIdx++));
                }
            }
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
            Map<Integer, NDArray> fwdValues = fillForwardValues(jaxpr, List.of(arg), null);
            NDArray value = fwdValues.get(jaxpr.outVars().get(0).id());
            NDArray seed = ones(value.shape(), floatDtypeOrDefault(value.dtype()));
            List<NDArray> grads = doBackward(jaxpr, fwdValues, List.of(seed), null);
            return new NDArray[]{value, grads.get(0)};
        };
    }

    private static List<NDArray> backward(Jaxpr jaxpr, List<NDArray> argValues) {
        Var outVar = jaxpr.outVars().get(0);
        Device targetDevice = argValues.isEmpty() ? Device.defaultDevice() : argValues.get(0).device();
        NDArray seed = ones(outVar.shape(), floatDtypeOrDefault(outVar.dtype())).to(targetDevice);
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
        return forwardInterpret(jaxpr, argValues, null);
    }

    public static List<NDArray> forwardInterpret(Jaxpr jaxpr, List<NDArray> argValues, Map<Integer, NDArray> parentValues) {
        Map<Integer, NDArray> values = fillForwardValues(jaxpr, argValues, parentValues);
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
        return backwardInterpret(jaxpr, argValues, seedGrads, null);
    }

    public static List<NDArray> backwardInterpret(Jaxpr jaxpr, List<NDArray> argValues, List<NDArray> seedGrads, Map<Integer, NDArray> parentValues) {
        Map<Integer, NDArray> values = fillForwardValues(jaxpr, argValues, parentValues);
        return doBackward(jaxpr, values, seedGrads, parentValues);
    }

    private static List<NDArray> doBackward(Jaxpr jaxpr, Map<Integer, NDArray> values, List<NDArray> seedGrads, Map<Integer, NDArray> parentValues) {
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
                List<NDArray> gIns = scanBackward(eq, inputs, gCarryOut, gYsOut, values);
                grads.merge(eq.inputs().get(0).id(), gIns.get(0), NDArray::add);
                grads.merge(eq.inputs().get(1).id(), gIns.get(1), NDArray::add);
                continue;
            }

            NDArray gOut = grads.get(eq.outputs().get(0).id());
            if (gOut == null) continue;

            List<NDArray> gIns = computeVJPs(eq.primitive(), eq, gOut, inputs, values);
            for (int j = 0; j < eq.inputs().size(); j++) {
                grads.merge(eq.inputs().get(j).id(), gIns.get(j), NDArray::add);
            }
        }

        List<NDArray> result = new ArrayList<>();
        for (int j = 0; j < jaxpr.inVars().size(); j++) {
            Var inVar = jaxpr.inVars().get(j);
            NDArray g = grads.get(inVar.id());
            if (g == null) {
                NDArray inputVal = values.get(inVar.id());
                Device dev = (inputVal != null) ? inputVal.device() : Device.defaultDevice();
                g = zeros(inVar.shape(), floatDtypeOrDefault(inVar.dtype())).to(dev);
            }
            result.add(g);
        }
        return result;
    }

    /** The forward pass shared by {@link #forwardInterpret} and {@link #backwardInterpret}. */
    private static Map<Integer, NDArray> fillForwardValues(Jaxpr jaxpr, List<NDArray> argValues, Map<Integer, NDArray> parentValues) {
        Device targetDevice = argValues.isEmpty() ? Device.defaultDevice() : argValues.get(0).device();
        Map<Integer, NDArray> values = new HashMap<>();
        if (parentValues != null) {
            values.putAll(parentValues);
        }
        for (Map.Entry<Integer, NDArray> entry : jaxpr.consts().entrySet()) {
            values.put(entry.getKey(), entry.getValue().to(targetDevice));
        }
        for (int i = 0; i < jaxpr.inVars().size(); i++) {
            values.put(jaxpr.inVars().get(i).id(), argValues.get(i));
        }

        for (Equation eq : jaxpr.equations()) {
            NDArray[] inputs = eq.inputs().stream().map(v -> values.get(v.id())).toArray(NDArray[]::new);
            if (eq.primitive() == Primitive.SCAN) {
                List<NDArray> outs = scanForward(eq, inputs, values);
                values.put(eq.outputs().get(0).id(), outs.get(0));
                values.put(eq.outputs().get(1).id(), outs.get(1));
            } else if (eq.primitive() == Primitive.RANDOM_SPLIT) {
                if (Tracer.current() != null) {
                    Tracer tracer = Tracer.current();
                    List<Var> outVars = new ArrayList<>();
                    for (int j = 0; j < eq.outputs().size(); j++) {
                        Var outVar = tracer.nextVar(new Shape(), DType.INT64);
                        outVars.add(outVar);
                        values.put(eq.outputs().get(j).id(), new TracedNDArray(outVar));
                    }
                    tracer.addEquation(new Equation(
                        List.of(((TracedNDArray) inputs[0]).getVar()),
                        outVars,
                        Primitive.RANDOM_SPLIT
                    ));
                } else {
                    PRNGKey[] subkeys = Random.splitEager(inputs[0], eq.outputs().size());
                    for (int j = 0; j < eq.outputs().size(); j++) {
                        values.put(eq.outputs().get(j).id(), subkeys[j].keyArray());
                    }
                }
            } else {
                NDArray out = executePrimitive(eq.primitive(), eq, inputs, values);
                values.put(eq.outputs().get(0).id(), out);
            }
        }
        return values;
    }

    /** Runs a {@code SCAN} equation's step function once per leading-axis slice of {@code xs}. */
    private static List<NDArray> scanForward(Equation eq, NDArray[] inputs, Map<Integer, NDArray> parentValues) {
        ScanMeta m = (ScanMeta) eq.metadata();
        NDArray carry = inputs[0];
        NDArray xs = inputs[1];
        int steps = xs.shape().dimensions()[0];
        Shape xStepShape = ScanUtil.dropLeadingDim(xs.shape());

        List<NDArray> ys = new ArrayList<>(steps);
        for (int t = 0; t < steps; t++) {
            NDArray xt = ScanUtil.sliceLeading(xs, t, xStepShape);
            List<NDArray> stepOut = forwardInterpret(m.stepFn(), List.of(carry, xt), parentValues);
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
    private static List<NDArray> scanBackward(Equation eq, NDArray[] inputs, NDArray gCarryFinal, NDArray gYsStacked, Map<Integer, NDArray> parentValues) {
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
            carry = forwardInterpret(m.stepFn(), List.of(carry, xIn[t]), parentValues).get(0);
        }

        NDArray gCarry = gCarryFinal != null ? gCarryFinal : zeros(initCarry.shape()).to(initCarry.device());
        NDArray[] gXs = new NDArray[steps];
        for (int t = steps - 1; t >= 0; t--) {
            NDArray gY = gYsStacked != null ? ScanUtil.sliceLeading(gYsStacked, t, yStepShape) : zeros(yStepShape).to(gCarry.device());
            List<NDArray> gStepIns = backwardInterpret(m.stepFn(), List.of(carryIn[t], xIn[t]), List.of(gCarry, gY), parentValues);
            gCarry = gStepIns.get(0);
            gXs[t] = gStepIns.get(1);
        }

        return List.of(gCarry, ScanUtil.stackLeading(java.util.Arrays.asList(gXs)));
    }

    private static NDArray executePrimitive(Primitive p, Equation eq, NDArray[] inputs, Map<Integer, NDArray> parentValues) {
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
                yield forwardInterpret(branch, List.of(inputs[1]), parentValues).get(0);
            }
            case WHILE -> {
                WhileMeta m = (WhileMeta) eq.metadata();
                NDArray state = inputs[0];
                while (forwardInterpret(m.condFn(), List.of(state), parentValues).get(0).toFloatArray()[0] != 0f) {
                    state = forwardInterpret(m.bodyFn(), List.of(state), parentValues).get(0);
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
                yield new ConcreteNDArray(ctx.collective.psum(ctx.deviceIndex, inputs[0].toFloatArray()), inputs[0].shape()).to(inputs[0].device());
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
                yield forwardInterpret(m.subJaxpr(), List.of(inputs[0]), parentValues).get(0);
            }
            case RESHAPE -> inputs[0].reshape(eq.outputs().get(0).shape());
            case TRANSPOSE -> {
                TransposeMeta m = (TransposeMeta) eq.metadata();
                yield inputs[0].transpose(m.axes());
            }
            case CONCAT -> {
                ConcatMeta m = (ConcatMeta) eq.metadata();
                yield Numpy.concatenateEager(java.util.Arrays.asList(inputs), m.axis(),
                    eq.outputs().get(0).shape(), inputs[0].dtype());
            }
            case PAD -> {
                PadMeta m = (PadMeta) eq.metadata();
                yield inputs[0].pad(m.padding());
            }
            case SCATTER_ADD -> Lax.scatterAddEager(inputs[0], inputs[1], inputs[2]);
            case FFT -> fftPackedForward(inputs[0], false);
            case IFFT -> fftPackedForward(inputs[0], true);
            case LINALG_SOLVE -> Linalg.solve(inputs[0], inputs[1]);
            case LINALG_SVD -> {
                // SVD returns (U, S, Vt) packed as 3 concatenated arrays.
                // For forward, just solve using the S as the result (eigenvalue-like scalar).
                // In practice the traced version packs S into a 1-D result.
                Linalg.Svd svd = Linalg.svd(inputs[0]);
                yield svd.sigma();
            }
            case LINALG_EIG -> {
                NDArray[] eig = Linalg.eig(inputs[0]);
                yield eig[0]; // real parts of eigenvalues
            }
            case CUSTOM_VJP -> {
                CustomVjpMeta m = (CustomVjpMeta) eq.metadata();
                yield m.fn().apply(inputs[0]);
            }
            case CONV2D -> {
                Conv2DMeta m = (Conv2DMeta) eq.metadata();
                yield Convolution.conv2dEager(inputs[0], inputs[1], m.strides(), m.padding());
            }
            case DEPTHWISE_CONV_2D -> {
                DepthwiseConv2DMeta m = (DepthwiseConv2DMeta) eq.metadata();
                yield Convolution.depthwiseConv2dEager(inputs[0], inputs[1], m.strides(), m.padding(), m.depthMultiplier());
            }
            case CONV_2D_TRANSPOSE -> {
                Conv2DTransposeMeta m = (Conv2DTransposeMeta) eq.metadata();
                yield Convolution.conv2dTransposeEager(inputs[0], inputs[1], m.strides(), m.padding());
            }
            case MAX_POOL_2D -> {
                Pool2DMeta m = (Pool2DMeta) eq.metadata();
                yield Convolution.maxPool2dEager(inputs[0], m.poolSize(), m.strides(), m.padding());
            }
            case AVG_POOL_2D -> {
                Pool2DMeta m = (Pool2DMeta) eq.metadata();
                yield Convolution.avgPool2dEager(inputs[0], m.poolSize(), m.strides(), m.padding());
            }
            case CONV3D -> {
                com.marmanis.jax4j.ir.Conv3DMeta m = (com.marmanis.jax4j.ir.Conv3DMeta) eq.metadata();
                yield Convolution.conv3dEager(inputs[0], inputs[1], m.strides(), m.padding());
            }
            case MAX_POOL_3D -> {
                com.marmanis.jax4j.ir.Pool3DMeta m = (com.marmanis.jax4j.ir.Pool3DMeta) eq.metadata();
                yield Convolution.maxPool3dEager(inputs[0], m.poolSize(), m.strides(), m.padding());
            }
            case AVG_POOL_3D -> {
                com.marmanis.jax4j.ir.Pool3DMeta m = (com.marmanis.jax4j.ir.Pool3DMeta) eq.metadata();
                yield Convolution.avgPool3dEager(inputs[0], m.poolSize(), m.strides(), m.padding());
            }
            case SQRT -> inputs[0].sqrt();
            case RSQRT -> inputs[0].rsqrt();
            case MAX_AXIS, MIN_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                yield p == Primitive.MAX_AXIS
                    ? inputs[0].max(m.axis(), m.keepDims())
                    : inputs[0].min(m.axis(), m.keepDims());
            }
            case SLICE -> {
                SliceMeta m = (SliceMeta) eq.metadata();
                yield inputs[0].slice(m.starts(), m.stops(), m.steps());
            }
            case MATMUL -> ConcreteNDArray.matmulEager(inputs[0], inputs[1]);
            case GRID_SAMPLE_2D -> {
                GridSample2DMeta m = (GridSample2DMeta) eq.metadata();
                yield GridSampling.gridSample2dForward(inputs[0], inputs[1], m.paddingMode());
            }
            case RANDOM_SEED -> {
                if (Tracer.current() != null) {
                    Tracer tracer = Tracer.current();
                    Var outVar = tracer.nextVar(new Shape(), DType.INT64);
                    long initialSeed = (Long) eq.metadata();
                    tracer.addEquation(new Equation(
                        List.of(),
                        List.of(outVar),
                        Primitive.RANDOM_SEED,
                        initialSeed
                    ));
                    yield new TracedNDArray(outVar);
                } else {
                    long initialSeed = (Long) eq.metadata();
                    long dynamicSeed = mix64(initialSeed ^ System.nanoTime());
                    yield new ConcreteNDArray(new long[]{dynamicSeed}, new Shape());
                }
            }
            case RANDOM_UNIFORM -> {
                RandomMeta.Uniform m = (RandomMeta.Uniform) eq.metadata();
                if (inputs[0] instanceof TracedNDArray traced) {
                    Tracer tracer = Tracer.current();
                    Var outVar = tracer.nextVar(m.shape(), DType.FLOAT32);
                    tracer.addEquation(new Equation(
                        List.of(traced.getVar()),
                        List.of(outVar),
                        Primitive.RANDOM_UNIFORM,
                        m
                    ));
                    yield new TracedNDArray(outVar);
                } else {
                    yield Random.uniformEager(inputs[0], m.shape(), m.lo(), m.hi());
                }
            }
            case RANDOM_NORMAL -> {
                RandomMeta.Normal m = (RandomMeta.Normal) eq.metadata();
                if (inputs[0] instanceof TracedNDArray traced) {
                    Tracer tracer = Tracer.current();
                    Var outVar = tracer.nextVar(m.shape(), DType.FLOAT32);
                    tracer.addEquation(new Equation(
                        List.of(traced.getVar()),
                        List.of(outVar),
                        Primitive.RANDOM_NORMAL,
                        m
                    ));
                    yield new TracedNDArray(outVar);
                } else {
                    yield Random.normalEager(inputs[0], m.shape());
                }
            }
            case RANDOM_BERNOULLI -> {
                RandomMeta.Bernoulli m = (RandomMeta.Bernoulli) eq.metadata();
                if (inputs[0] instanceof TracedNDArray traced) {
                    Tracer tracer = Tracer.current();
                    Var outVar = tracer.nextVar(m.shape(), DType.BOOL);
                    tracer.addEquation(new Equation(
                        List.of(traced.getVar()),
                        List.of(outVar),
                        Primitive.RANDOM_BERNOULLI,
                        m
                    ));
                    yield new TracedNDArray(outVar);
                } else {
                    yield Random.bernoulliEager(inputs[0], m.p(), m.shape());
                }
            }
            case RANDOM_PERMUTATION -> {
                RandomMeta.Permutation m = (RandomMeta.Permutation) eq.metadata();
                if (inputs[0] instanceof TracedNDArray traced) {
                    Tracer tracer = Tracer.current();
                    Var outVar = tracer.nextVar(new Shape(m.n()), DType.INT32);
                    tracer.addEquation(new Equation(
                        List.of(traced.getVar()),
                        List.of(outVar),
                        Primitive.RANDOM_PERMUTATION,
                        m
                    ));
                    yield new TracedNDArray(outVar);
                } else {
                    yield Random.permutationEager(inputs[0], m.n());
                }
            }
            default -> throw new UnsupportedOperationException(p.toString());
        };
    }

    private static List<NDArray> computeVJPs(Primitive p, Equation eq, NDArray gOut, NDArray[] inputs, Map<Integer, NDArray> parentValues) {
        return switch (p) {
            case ADD -> List.of(broadcastLike(gOut, inputs[0]), broadcastLike(gOut, inputs[1]));
            case SUB -> List.of(broadcastLike(gOut, inputs[0]), broadcastLike(gOut.mul(minusOne(gOut.shape(), gOut.dtype(), gOut.device())), inputs[1]));
            case MUL -> List.of(broadcastLike(gOut.mul(inputs[1]), inputs[0]), broadcastLike(gOut.mul(inputs[0]), inputs[1]));
            case DIV -> {
                // d/da (a/b) = 1/b ; d/db (a/b) = -a/b^2
                NDArray a = inputs[0];
                NDArray b = inputs[1];
                NDArray gA = broadcastLike(gOut.div(b), a);
                NDArray gB = broadcastLike(gOut.mul(a).div(b).div(b).mul(minusOne(gOut.shape(), gOut.dtype(), gOut.device())), b);
                yield List.of(gA, gB);
            }
            case MEAN -> {
                double n = inputs[0].shape().size();
                NDArray gIn = broadcastLike(gOut.div(scalar(n, inputs[0].dtype(), gOut.device())), inputs[0]);
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
                NDArray scaled = gOut.div(scalar(axisSize, gOut.dtype(), gOut.device()));
                yield List.of(axisBroadcastLike(scaled, inputs[0].shape(), m.axis()));
            }
            case EXP -> List.of(gOut.mul(inputs[0].exp()));
            case LOG -> List.of(gOut.div(inputs[0]));
            case SIN -> List.of(gOut.mul(inputs[0].cos()));
            case COS -> List.of(gOut.mul(inputs[0].sin()).mul(minusOne(gOut.shape(), gOut.dtype(), gOut.device())));
            case TANH -> {
                // d/dx tanh(x) = 1 - tanh(x)^2
                NDArray t = inputs[0].tanh();
                yield List.of(gOut.mul(ones(t.shape(), t.dtype(), t.device()).sub(t.mul(t))));
            }
            case RELU -> List.of(reluGrad(gOut, inputs[0]));
            case SIGMOID -> {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                NDArray s = inputs[0].sigmoid();
                yield List.of(gOut.mul(s.mul(ones(s.shape(), s.dtype(), s.device()).sub(s))));
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
                NDArray gOperand = backwardInterpret(branch, List.of(inputs[1]), List.of(gOut), parentValues).get(0);
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
            case RANDOM_SEED ->
                List.of();
            case RANDOM_SPLIT, RANDOM_UNIFORM, RANDOM_NORMAL, RANDOM_BERNOULLI, RANDOM_PERMUTATION ->
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
                    ctx.collective.psum(ctx.deviceIndex, gOut.toFloatArray()), gOut.shape()).to(gOut.device()));
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
            case RESHAPE -> List.of(gOut.reshape(inputs[0].shape()));
            case TRANSPOSE -> {
                TransposeMeta m = (TransposeMeta) eq.metadata();
                yield List.of(gOut.transpose(invertPerm(m.axes())));
            }
            case CONCAT -> {
                ConcatMeta m = (ConcatMeta) eq.metadata();
                yield splitAlongAxis(gOut, inputs, m.axis());
            }
            case PAD -> {
                PadMeta m = (PadMeta) eq.metadata();
                yield List.of(unpad(gOut, m.padding(), inputs[0].shape()));
            }
            case SCATTER_ADD -> {
                // VJP wrt target: passthrough (identity on gOut)
                // VJP wrt indices: zero (non-differentiable)
                // VJP wrt updates: gather gOut at indices
                NDArray gTarget = broadcastLike(gOut, inputs[0]);
                NDArray gIndices = zerosLike(inputs[1]);
                NDArray gUpdates = Numpy.takeEager(gOut, inputs[1]);
                yield List.of(gTarget, gIndices, gUpdates);
            }
            case FFT -> List.of(fftPackedForward(gOut, true));   // grad of FFT = IFFT
            case IFFT -> List.of(fftPackedForward(gOut, false));  // grad of IFFT = FFT
            case LINALG_SOLVE -> {
                // A x = b. g_b = A^T^{-1} g_x = solve(A^T, g_x)
                // g_A = -outer(g_b, x)
                NDArray A = inputs[0];
                NDArray x = eq.outputs().get(0) != null ? Linalg.solve(A, inputs[1]) : inputs[1]; // recompute x
                NDArray gB = solveTransposeGrad(A, gOut);
                NDArray gA = outerNeg(gB, x);
                yield List.of(gA, gB);
            }
            case LINALG_SVD, LINALG_EIG -> {
                // Simplified: zero gradient (not fully differentiable in this stub)
                yield List.of(zerosLike(inputs[0]));
            }
            case CUSTOM_VJP -> {
                CustomVjpMeta m = (CustomVjpMeta) eq.metadata();
                yield List.of(m.vjpFn().apply(inputs[0], gOut));
            }
            case CONV2D -> {
                Conv2DMeta m = (Conv2DMeta) eq.metadata();
                NDArray[] gs = Convolution.conv2dBackward(inputs[0], inputs[1], gOut, m.strides(), m.padding());
                yield List.of(gs[0], gs[1]);
            }
            case DEPTHWISE_CONV_2D -> {
                DepthwiseConv2DMeta m = (DepthwiseConv2DMeta) eq.metadata();
                NDArray[] gs = Convolution.depthwiseConv2dBackward(inputs[0], inputs[1], gOut,
                    m.strides(), m.padding(), m.depthMultiplier());
                yield List.of(gs[0], gs[1]);
            }
            case CONV_2D_TRANSPOSE -> {
                Conv2DTransposeMeta m = (Conv2DTransposeMeta) eq.metadata();
                NDArray[] gs = Convolution.conv2dTransposeBackward(inputs[0], inputs[1], gOut, m.strides(), m.padding());
                yield List.of(gs[0], gs[1]);
            }
            case MAX_POOL_2D -> {
                Pool2DMeta m = (Pool2DMeta) eq.metadata();
                yield List.of(Convolution.maxPool2dBackward(inputs[0], gOut, m.poolSize(), m.strides(), m.padding()));
            }
            case AVG_POOL_2D -> {
                Pool2DMeta m = (Pool2DMeta) eq.metadata();
                yield List.of(Convolution.avgPool2dBackward(gOut, inputs[0].shape().dimensions(), m.poolSize(), m.strides(), m.padding()));
            }
            case SQRT -> {
                // d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 * rsqrt(x)
                NDArray half = scalar(0.5, gOut.dtype(), gOut.device());
                yield List.of(gOut.mul(half).div(inputs[0].sqrt()));
            }
            case RSQRT -> {
                // d/dx (x^-1/2) = -0.5 * x^-3/2 = -0.5 * rsqrt(x)^3
                NDArray r = inputs[0].rsqrt();
                NDArray coeff = scalar(-0.5, gOut.dtype(), gOut.device());
                yield List.of(gOut.mul(coeff).mul(r).mul(r).mul(r));
            }
            case MAX_AXIS, MIN_AXIS -> {
                AxisMeta m = (AxisMeta) eq.metadata();
                yield List.of(maxMinAxisGrad(gOut, inputs[0], m.axis(), m.keepDims(), p == Primitive.MAX_AXIS));
            }
            case SLICE -> {
                SliceMeta m = (SliceMeta) eq.metadata();
                yield List.of(sliceBackward(gOut, inputs[0].shape(), m));
            }
            case MATMUL -> {
                NDArray a = inputs[0];
                NDArray b = inputs[1];
                NDArray bT = swapLastTwo(b);
                NDArray aT = swapLastTwo(a);
                NDArray gA = matmulReduceBatch(ConcreteNDArray.matmulEager(gOut, bT), a.shape());
                NDArray gB = matmulReduceBatch(ConcreteNDArray.matmulEager(aT, gOut), b.shape());
                yield List.of(gA, gB);
            }
            case GRID_SAMPLE_2D -> {
                GridSample2DMeta m = (GridSample2DMeta) eq.metadata();
                NDArray gIn = GridSampling.gridSample2dBackwardInput(inputs[0], inputs[1], gOut, m.paddingMode());
                yield List.of(gIn, zerosLike(inputs[1]));
            }
            default -> throw new UnsupportedOperationException(p.toString());
        };
    }

    /** Transposes the last two axes of x, keeping leading batch axes in place. */
    private static NDArray swapLastTwo(NDArray x) {
        int rank = x.shape().rank();
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) axes[i] = i;
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return x.transpose(axes);
    }

    /**
     * Reduces broadcast batch axes: after batched matmul with broadcasting, the
     * output batch shape may exceed one operand's. Sum-reduce over any leading
     * axes that were size-1 in the operand.
     */
    private static NDArray matmulReduceBatch(NDArray g, Shape targetShape) {
        int[] gDims = g.shape().dimensions();
        int[] tDims = targetShape.dimensions();
        if (java.util.Arrays.equals(gDims, tDims)) return g;
        NDArray current = g;
        int gRank = gDims.length;
        int tRank = tDims.length;
        // Sum-reduce leading axes that don't exist in target.
        while (current.shape().rank() > tRank) {
            current = current.sum(0, false);
        }
        // For remaining aligned batch axes (all but the last two), reduce where target has size 1.
        int[] cDims = current.shape().dimensions();
        for (int i = 0; i < cDims.length - 2; i++) {
            if (tDims[i] == 1 && cDims[i] > 1) {
                current = current.sum(i, true);
            }
        }
        if (!current.shape().equals(targetShape)) {
            current = current.reshape(targetShape);
        }
        return current;
    }

    /**
     * VJP for slice: zero-pad the gradient back to the original input shape, placing
     * each output element at position {@code starts[i] + coord * steps[i]}.
     */
    private static NDArray sliceBackward(NDArray gOut, Shape origShape, SliceMeta m) {
        int rank = origShape.rank();
        int[] origDims = origShape.dimensions();
        int[] gDims = gOut.shape().dimensions();
        int[] inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * origDims[i + 1];

        int gSize = (int) gOut.shape().size();
        if (gOut.dtype() == DType.FLOAT64) {
            double[] g = gOut.toDoubleArray();
            double[] out = new double[(int) origShape.size()];
            for (int gFlat = 0; gFlat < gSize; gFlat++) {
                int rem = gFlat;
                int inFlat = 0;
                for (int i = rank - 1; i >= 0; i--) {
                    int coord = rem % gDims[i];
                    rem /= gDims[i];
                    inFlat += (m.starts()[i] + coord * m.steps()[i]) * inStrides[i];
                }
                out[inFlat] = g[gFlat];
            }
            return new ConcreteNDArray(out, origShape, gOut.device());
        }
        float[] g = gOut.toFloatArray();
        float[] out = new float[(int) origShape.size()];
        for (int gFlat = 0; gFlat < gSize; gFlat++) {
            int rem = gFlat;
            int inFlat = 0;
            for (int i = rank - 1; i >= 0; i--) {
                int coord = rem % gDims[i];
                rem /= gDims[i];
                inFlat += (m.starts()[i] + coord * m.steps()[i]) * inStrides[i];
            }
            out[inFlat] = g[gFlat];
        }
        return new ConcreteNDArray(out, origShape, gOut.dtype(), gOut.device());
    }

    /**
     * VJP for max/min axis reduction: gradient flows to argmax/argmin positions
     * within each reduction slice; on ties, gradient is split equally.
     */
    private static NDArray maxMinAxisGrad(NDArray gOut, NDArray x, int axis, boolean keepDims, boolean isMax) {
        int norm = x.shape().normalizeAxis(axis);
        int[] dims = x.shape().dimensions();
        int axisSize = dims[norm];
        int outerSize = 1;
        for (int i = 0; i < norm; i++) outerSize *= dims[i];
        int innerSize = 1;
        for (int i = norm + 1; i < dims.length; i++) innerSize *= dims[i];

        if (x.dtype() == DType.FLOAT64) {
            double[] xv = x.toDoubleArray();
            double[] gv = gOut.toDoubleArray();
            double[] out = new double[(int) x.shape().size()];
            for (int o = 0; o < outerSize; o++) {
                for (int inr = 0; inr < innerSize; inr++) {
                    double best = xv[o * axisSize * innerSize + inr];
                    for (int a = 1; a < axisSize; a++) {
                        double v = xv[o * axisSize * innerSize + a * innerSize + inr];
                        if (isMax ? v > best : v < best) best = v;
                    }
                    int tieCount = 0;
                    for (int a = 0; a < axisSize; a++) {
                        double v = xv[o * axisSize * innerSize + a * innerSize + inr];
                        if (v == best) tieCount++;
                    }
                    double gVal = gv[o * innerSize + inr] / tieCount;
                    for (int a = 0; a < axisSize; a++) {
                        double v = xv[o * axisSize * innerSize + a * innerSize + inr];
                        if (v == best) out[o * axisSize * innerSize + a * innerSize + inr] = gVal;
                    }
                }
            }
            return new ConcreteNDArray(out, x.shape(), x.device());
        }
        float[] xv = x.toFloatArray();
        float[] gv = gOut.toFloatArray();
        float[] out = new float[(int) x.shape().size()];
        for (int o = 0; o < outerSize; o++) {
            for (int inr = 0; inr < innerSize; inr++) {
                float best = xv[o * axisSize * innerSize + inr];
                for (int a = 1; a < axisSize; a++) {
                    float v = xv[o * axisSize * innerSize + a * innerSize + inr];
                    if (isMax ? v > best : v < best) best = v;
                }
                int tieCount = 0;
                for (int a = 0; a < axisSize; a++) {
                    float v = xv[o * axisSize * innerSize + a * innerSize + inr];
                    if (v == best) tieCount++;
                }
                float gVal = gv[o * innerSize + inr] / tieCount;
                for (int a = 0; a < axisSize; a++) {
                    float v = xv[o * axisSize * innerSize + a * innerSize + inr];
                    if (v == best) out[o * axisSize * innerSize + a * innerSize + inr] = gVal;
                }
            }
        }
        return new ConcreteNDArray(out, x.shape(), x.dtype(), x.device());
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
        return new ConcreteNDArray(gTable, tableShape, gOut.dtype(), gOut.device());
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
            return List.of(broadcastLike(new ConcreteNDArray(gA, outShape, a.device()), a),
                            broadcastLike(new ConcreteNDArray(gB, outShape, b.device()), b));
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
        return List.of(broadcastLike(new ConcreteNDArray(gA, outShape, gOut.dtype(), a.device()), a),
                        broadcastLike(new ConcreteNDArray(gB, outShape, gOut.dtype(), b.device()), b));
    }

    /**
     * Reconciles a gradient {@code g} with {@code target}'s original shape.
     * If {@code target} is smaller (it was broadcast up during the forward pass),
     * sums the contributions back down. If {@code target} is larger (e.g. the
     * MEAN/SUM vjp seeds a scalar gradient that must spread across every input
     * element), expands {@code g} back out via broadcasting.
     */
    private static NDArray onesForShape(Shape shape, DType dtype, Device device) {
        int n = (int) shape.size();
        if (dtype == DType.FLOAT64) {
            double[] ones = new double[n];
            java.util.Arrays.fill(ones, 1.0);
            return new ConcreteNDArray(ones, shape, device);
        }
        float[] ones = new float[n];
        java.util.Arrays.fill(ones, 1.0f);
        return new ConcreteNDArray(ones, shape, DType.FLOAT32, device);
    }

    private static NDArray broadcastLike(NDArray g, NDArray target) {
        Shape gShape = g.shape();
        Shape tShape = target.shape();
        if (gShape.equals(tShape)) return g;

        if (Tracer.current() != null || g instanceof TracedNDArray || target instanceof TracedNDArray) {
            if (gShape.size() == tShape.size()) {
                return g.reshape(tShape);
            }
            if (gShape.size() < tShape.size()) {
                NDArray ones = onesForShape(tShape, g.dtype(), g.device());
                return g.mul(ones);
            } else {
                NDArray current = g;
                int[] gDims = gShape.dimensions();
                int[] tDims = tShape.dimensions();
                int gRank = gDims.length;
                int tRank = tDims.length;

                for (int i = 0; i < gRank; i++) {
                    int tIdx = i - (gRank - tRank);
                    if (tIdx < 0) {
                        current = current.sum(i, true);
                    } else if (tDims[tIdx] == 1 && gDims[i] > 1) {
                        current = current.sum(i, true);
                    }
                }
                return current.reshape(tShape);
            }
        }

        if (g.dtype() == DType.FLOAT64) {
            if (gShape.size() == tShape.size()) {
                return new ConcreteNDArray(g.toDoubleArray(), tShape, target.device());
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
            return new ConcreteNDArray(result, tShape, target.device());
        }

        if (gShape.size() == tShape.size()) {
            // Same total size, different rank/shape: a pure reshape, no reduction needed.
            return new ConcreteNDArray(g.toFloatArray(), tShape, g.dtype(), target.device());
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
        return new ConcreteNDArray(result, tShape, g.dtype(), target.device());
    }

    /** d/dx relu(x) = 1 if x > 0 else 0; no comparison primitive exists, so this drops to raw arrays. */
    private static NDArray reluGrad(NDArray gOut, NDArray x) {
        if (x.dtype() == DType.FLOAT64) {
            double[] g = gOut.toDoubleArray();
            double[] xv = x.toDoubleArray();
            double[] result = new double[xv.length];
            for (int i = 0; i < xv.length; i++) result[i] = xv[i] > 0 ? g[i] : 0.0;
            return new ConcreteNDArray(result, x.shape(), x.device());
        }
        float[] g = gOut.toFloatArray();
        float[] xv = x.toFloatArray();
        float[] result = new float[xv.length];
        for (int i = 0; i < xv.length; i++) result[i] = xv[i] > 0 ? g[i] : 0f;
        return new ConcreteNDArray(result, x.shape(), gOut.dtype(), x.device());
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

            NDArray outPrimal = executePrimitive(eq.primitive(), eq, inPrimals, primals);
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
            case RESHAPE -> {
                return tangents[0].reshape(primalOut.shape());
            }
            case TRANSPOSE -> {
                TransposeMeta m = (TransposeMeta) eq.metadata();
                return tangents[0].transpose(m.axes());
            }
            case CONCAT -> {
                ConcatMeta m = (ConcatMeta) eq.metadata();
                return Numpy.concatenateEager(java.util.Arrays.asList(tangents), m.axis(), primalOut.shape(), primalOut.dtype());
            }
            case PAD -> {
                PadMeta m = (PadMeta) eq.metadata();
                return tangents[0].pad(m.padding());
            }
            case SCATTER_ADD -> {
                // scatter_add is linear in target and updates; indices are non-differentiable
                return Lax.scatterAddEager(tangents[0], primals[1], tangents[2]);
            }
            case FFT -> { return fftPackedForward(tangents[0], false); }
            case IFFT -> { return fftPackedForward(tangents[0], true); }
            case LINALG_SOLVE, LINALG_SVD, LINALG_EIG, CUSTOM_VJP -> {
                return zeros(shape, dtype);
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
        return scalar(v, dtype, Device.defaultDevice());
    }

    private static NDArray scalar(double v, DType dtype, Device device) {
        return switch (dtype) {
            case FLOAT32 -> new ConcreteNDArray(new float[]{(float) v}, new Shape(1), dtype, device);
            case FLOAT64 -> new ConcreteNDArray(new double[]{v}, new Shape(1), device);
            default -> throw new IllegalArgumentException("scalar() only supports floating dtypes, got " + dtype);
        };
    }

    private static NDArray ones(Shape shape, DType dtype) {
        return ones(shape, dtype, Device.defaultDevice());
    }

    private static NDArray ones(Shape shape, DType dtype, Device device) {
        return switch (dtype) {
            case FLOAT32 -> {
                float[] data = new float[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = 1.0f;
                yield new ConcreteNDArray(data, shape, dtype, device);
            }
            case FLOAT64 -> {
                double[] data = new double[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = 1.0;
                yield new ConcreteNDArray(data, shape, device);
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
        return zeros(x.shape(), floatDtypeOrDefault(x.dtype())).to(x.device());
    }

    /** Returns the inverse permutation of {@code perm}: {@code inv[perm[i]] = i}. */
    private static int[] invertPerm(int[] perm) {
        int[] inv = new int[perm.length];
        for (int i = 0; i < perm.length; i++) inv[perm[i]] = i;
        return inv;
    }

    /**
     * Splits {@code x} along {@code axis} into chunks matching the axis-size of
     * each corresponding input array. Returns one gradient chunk per input.
     */
    private static List<NDArray> splitAlongAxis(NDArray x, NDArray[] inputs, int axis) {
        List<NDArray> result = new java.util.ArrayList<>(inputs.length);
        int offset = 0;
        for (NDArray input : inputs) {
            int size = input.shape().dimensions()[axis];
            result.add(Numpy.sliceAxis(x, axis, offset, offset + size));
            offset += size;
        }
        return result;
    }

    /**
     * Slices away the padding from a padded gradient, recovering the original shape.
     */
    private static NDArray unpad(NDArray x, int[][] padding, Shape origShape) {
        // For each dim i, we want coords in [padding[i][0], padding[i][0] + origShape.dimensions()[i])
        // This is equivalent to sliceAxis applied per dimension.
        NDArray result = x;
        int[] dims = x.shape().dimensions();
        int rank = dims.length;
        // Work from last axis to first so slicing dimensions stay consistent
        for (int i = rank - 1; i >= 0; i--) {
            int before = padding[i][0];
            int origSize = origShape.dimensions()[i];
            result = Numpy.sliceAxis(result, i, before, before + origSize);
        }
        return result;
    }

    /**
     * FFT autodiff helper: applies forward or inverse FFT to a packed [N, 2] complex array
     * where {@code arr[k, 0]} is the real part and {@code arr[k, 1]} the imaginary part.
     * When {@code inverse=true} runs the normalized IFFT.
     * Package-visible so Vmap can reuse the same kernel.
     */
    static NDArray fftPackedExec(NDArray arr, boolean inverse) { return fftPackedForward(arr, inverse); }

    private static NDArray fftPackedForward(NDArray arr, boolean inverse) {
        int[] dims = arr.shape().dimensions();
        int n = dims[0]; // number of frequency / time bins
        // Split into re and im
        NDArray re = Numpy.sliceAxis(arr, 1, 0, 1).reshape(n);
        NDArray im = Numpy.sliceAxis(arr, 1, 1, 2).reshape(n);
        NDArray[] out;
        if (arr.dtype() == DType.FLOAT64) {
            out = inverse ? Fft.ifft(re, im) : Fft.fft(re, im);
        } else {
            out = inverse ? Fft.ifft(re, im) : Fft.fft(re, im);
        }
        // Pack back into [N, 2]
        NDArray outRe = out[0].reshape(new com.marmanis.jax4j.core.Shape(n, 1));
        NDArray outIm = out[1].reshape(new com.marmanis.jax4j.core.Shape(n, 1));
        return Numpy.concatenateEager(List.of(outRe, outIm), 1,
            new com.marmanis.jax4j.core.Shape(n, 2), arr.dtype());
    }

    /** Computes {@code -outer(a, b)} = {@code -a[:, None] * b[None, :]} for 1-D vectors. */
    private static NDArray outerNeg(NDArray a, NDArray b) {
        int m = (int) a.shape().size();
        int n = (int) b.shape().size();
        if (a.dtype() == DType.FLOAT64) {
            double[] av = a.toDoubleArray();
            double[] bv = b.toDoubleArray();
            double[] out = new double[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    out[i * n + j] = -av[i] * bv[j];
            return new ConcreteNDArray(out, new Shape(m, n), a.device());
        }
        float[] av = a.toFloatArray();
        float[] bv = b.toFloatArray();
        float[] out = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                out[i * n + j] = -av[i] * bv[j];
        return new ConcreteNDArray(out, new Shape(m, n), a.dtype(), a.device());
    }

    /**
     * Solves {@code A^T g = gOut} via LU factorization of A^T.
     * Used in the LINALG_SOLVE VJP: {@code g_b = solve(A^T, g_x)}.
     */
    private static NDArray solveTransposeGrad(NDArray A, NDArray gOut) {
        // Transpose A and solve the system
        NDArray At = A.transpose(1, 0);
        return Linalg.solve(At, gOut);
    }

    private static NDArray minusOne(Shape shape, DType dtype) {
        return minusOne(shape, dtype, Device.defaultDevice());
    }

    private static NDArray minusOne(Shape shape, DType dtype, Device device) {
        return switch (dtype) {
            case FLOAT32 -> {
                float[] data = new float[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = -1.0f;
                yield new ConcreteNDArray(data, shape, dtype, device);
            }
            case FLOAT64 -> {
                double[] data = new double[(int) shape.size()];
                for (int i = 0; i < data.length; i++) data[i] = -1.0;
                yield new ConcreteNDArray(data, shape, device);
            }
            default -> throw new IllegalArgumentException("minusOne() only supports floating dtypes, got " + dtype);
        };
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }
}
