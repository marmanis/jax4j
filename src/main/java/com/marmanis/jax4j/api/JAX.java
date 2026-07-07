package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.ir.CheckpointMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.PmapMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Main entry point for JAX transformations.
 */
public class JAX {

    /**
     * Traces a function and returns its Jaxpr.
     */
    public static Jaxpr make_jaxpr(Function<NDArray, NDArray> fn, NDArray arg) {
        Tracer.start();
        try {
            Var inVar = Tracer.current().nextVar(arg.shape(), arg.dtype());
            NDArray result = fn.apply(new TracedNDArray(inVar));
            Var outVar = ((TracedNDArray) result).getVar();
            return Tracer.stop(List.of(inVar), List.of(outVar));
        } catch (RuntimeException | Error e) {
            Tracer.abort();
            throw e;
        }
    }

    public static Jaxpr make_jaxpr(BiFunction<NDArray, NDArray, NDArray> fn, NDArray arg1, NDArray arg2) {
        Tracer.start();
        try {
            Var v1 = Tracer.current().nextVar(arg1.shape(), arg1.dtype());
            Var v2 = Tracer.current().nextVar(arg2.shape(), arg2.dtype());
            NDArray result = fn.apply(new TracedNDArray(v1), new TracedNDArray(v2));
            Var outVar = ((TracedNDArray) result).getVar();
            return Tracer.stop(List.of(v1, v2), List.of(outVar));
        } catch (RuntimeException | Error e) {
            Tracer.abort();
            throw e;
        }
    }

    /**
     * Wraps {@code fn} so it is traced once per distinct input signature
     * (shape + dtype) and re-executed via {@link Grad#forwardInterpret}
     * on subsequent calls. Repeated invocations with the same signature
     * hit the cache and skip the tracing overhead.
     *
     * <p>Behaviour:
     * <ul>
     *   <li>First call at a given {@code (shape, dtype)}: trace via
     *       {@link #make_jaxpr}, store the resulting {@link Jaxpr},
     *       execute via {@link Grad#forwardInterpret}.</li>
     *   <li>Later calls at the same signature: skip tracing, execute the
     *       cached Jaxpr directly against the concrete input.</li>
     *   <li>Nested tracing (call inside another
     *       {@link Tracer#current() active tracer}): fall through to
     *       eager execution so the outer trace still records the ops —
     *       matching the standard JAX behaviour where {@code jit} is
     *       transparent under {@code grad}/{@code vmap}.</li>
     * </ul>
     *
     * <p>This is a <em>trace cache</em>, not a compilation cache — no
     * TornadoVM kernel is generated. Wrap with {@link #jitGrad} for
     * gradient-of-a-traced-function caching.
     */
    public static Function<NDArray, NDArray> jit(Function<NDArray, NDArray> fn) {
        java.util.concurrent.ConcurrentHashMap<TraceSignature, Jaxpr> cache =
            new java.util.concurrent.ConcurrentHashMap<>();
        return (arg) -> {
            if (Tracer.current() != null) {
                // Nested inside another trace — stay transparent.
                return fn.apply(arg);
            }
            TraceSignature key = new TraceSignature(arg.shape(), arg.dtype());
            Jaxpr jaxpr = cache.computeIfAbsent(key, k -> make_jaxpr(fn, arg));
            return Grad.forwardInterpret(jaxpr, List.of(arg)).get(0);
        };
    }

    /**
     * Wraps {@code fn} so its gradient is traced once per distinct input
     * signature and re-evaluated via {@link Grad#backwardInterpret} on
     * subsequent calls — the {@link #grad} analogue of {@link #jit}.
     * Semantics match {@link #grad}; only the tracing cost is amortized.
     *
     * <p>Uses the same signature-keyed cache as {@link #jit}. Nested
     * inside another tracer, falls through to
     * {@code Grad.grad(fn).apply(arg)} eagerly.
     */
    public static Function<NDArray, NDArray> jitGrad(Function<NDArray, NDArray> fn) {
        java.util.concurrent.ConcurrentHashMap<TraceSignature, Jaxpr> cache =
            new java.util.concurrent.ConcurrentHashMap<>();
        return (arg) -> {
            if (Tracer.current() != null) {
                return Grad.grad(fn).apply(arg);
            }
            TraceSignature key = new TraceSignature(arg.shape(), arg.dtype());
            Jaxpr jaxpr = cache.computeIfAbsent(key, k -> make_jaxpr(fn, arg));
            // Seed backward with ones of the appropriate shape/dtype.
            com.marmanis.jax4j.ir.Var outVar = jaxpr.outVars().get(0);
            NDArray seed = onesFor(outVar);
            return Grad.backwardInterpret(jaxpr, List.of(arg), List.of(seed)).get(0);
        };
    }

    /**
     * Caching analogue of {@link #gradTree}. Traces {@code fn} once per
     * distinct pytree signature (the list of {@code (shape, dtype)} pairs
     * of the leaves in flatten order) and reuses the traced Jaxpr for
     * subsequent backward passes.
     *
     * <p>Use this when the "constants" of a scalar {@link #grad} closure
     * actually vary between calls: pack every input into a pytree so the
     * trace captures them all as inputs (not baked-in constants). Then
     * repeated calls with different input <em>values</em> but the same
     * <em>signature</em> hit the cache.
     *
     * <p>Nested inside another tracer, falls through to
     * {@link Grad#gradTree(Function)} eagerly.
     */
    public static Function<PyTree, PyTree> jitGradTree(Function<PyTree, NDArray> fn) {
        java.util.concurrent.ConcurrentHashMap<TreeSignature, Jaxpr> cache =
            new java.util.concurrent.ConcurrentHashMap<>();
        return (paramsTree) -> {
            if (Tracer.current() != null) {
                return Grad.gradTree(fn).apply(paramsTree);
            }
            List<NDArray> leaves = com.marmanis.jax4j.pytree.PyTrees.flatten(paramsTree);
            TreeSignature key = TreeSignature.of(leaves);
            Jaxpr jaxpr = cache.computeIfAbsent(key, k -> traceGradTree(fn, paramsTree, leaves));
            com.marmanis.jax4j.ir.Var outVar = jaxpr.outVars().get(0);
            NDArray seed = onesFor(outVar);
            List<NDArray> grads = Grad.backwardInterpret(jaxpr, leaves, List.of(seed));
            return com.marmanis.jax4j.pytree.PyTrees.unflatten(paramsTree, grads);
        };
    }

    /** Trace {@code fn} with fresh {@link TracedNDArray}s for each leaf. */
    private static Jaxpr traceGradTree(Function<PyTree, NDArray> fn, PyTree structureSample,
                                       List<NDArray> leaves) {
        Tracer.start();
        try {
            java.util.List<com.marmanis.jax4j.ir.Var> inVars = new java.util.ArrayList<>();
            java.util.List<NDArray> tracedLeaves = new java.util.ArrayList<>();
            for (NDArray leaf : leaves) {
                com.marmanis.jax4j.ir.Var v = Tracer.current().nextVar(leaf.shape(), leaf.dtype());
                inVars.add(v);
                tracedLeaves.add(new TracedNDArray(v));
            }
            PyTree tracedTree = com.marmanis.jax4j.pytree.PyTrees.unflatten(structureSample, tracedLeaves);
            NDArray result = fn.apply(tracedTree);
            com.marmanis.jax4j.ir.Var outVar = ((TracedNDArray) result).getVar();
            return Tracer.stop(inVars, List.of(outVar));
        } catch (RuntimeException | Error e) {
            Tracer.abort();
            throw e;
        }
    }

    /** Cache key for {@link #jit} / {@link #jitGrad}: a shape and dtype pair. */
    private record TraceSignature(com.marmanis.jax4j.core.Shape shape, DType dtype) {}

    /**
     * Cache key for {@link #jitGradTree}: the ordered list of leaf
     * signatures. Two trees are cache-compatible iff their flatten
     * order gives the same list of {@code (shape, dtype)} pairs.
     */
    private record TreeSignature(java.util.List<TraceSignature> leafSignatures) {
        static TreeSignature of(java.util.List<NDArray> leaves) {
            java.util.List<TraceSignature> sigs = new java.util.ArrayList<>(leaves.size());
            for (NDArray leaf : leaves) sigs.add(new TraceSignature(leaf.shape(), leaf.dtype()));
            return new TreeSignature(java.util.List.copyOf(sigs));
        }
    }

    /** Ones NDArray matching a Jaxpr output variable's shape and (floating) dtype. */
    private static NDArray onesFor(com.marmanis.jax4j.ir.Var outVar) {
        int n = (int) outVar.shape().size();
        DType dtype = outVar.dtype();
        if (dtype == DType.FLOAT64) {
            double[] ones = new double[n];
            java.util.Arrays.fill(ones, 1.0);
            return new com.marmanis.jax4j.core.ConcreteNDArray(ones, outVar.shape());
        }
        // Default to FLOAT32 for anything else (matches the existing Grad seed).
        float[] ones = new float[n];
        java.util.Arrays.fill(ones, 1.0f);
        return new com.marmanis.jax4j.core.ConcreteNDArray(ones, outVar.shape());
    }

    /**
     * Returns a function that computes both {@code fn(x)} and its gradient in a
     * single forward+backward pass, mirroring {@code jax.value_and_grad}.
     * The returned array is {@code [value, grad]}.
     */
    public static Function<NDArray, NDArray[]> value_and_grad(Function<NDArray, NDArray> fn) {
        return Grad.value_and_grad(fn);
    }

    /**
     * Wraps {@code fn} so that its intermediate activations are not retained
     * during the forward pass; they are recomputed during the backward pass
     * (rematerialization). Mirrors {@code jax.checkpoint} / {@code jax.remat}.
     *
     * <p>Outside a gradient context (eager mode) this is a no-op. Inside
     * {@code JAX.grad}, a {@link Primitive#CHECKPOINT} equation is emitted so
     * the backward pass re-runs the forward through {@code fn} from scratch
     * rather than reading cached activations — useful when memory is the
     * bottleneck and recomputation is cheap relative to storage.
     */
    public static Function<NDArray, NDArray> checkpoint(Function<NDArray, NDArray> fn) {
        return (arg) -> {
            if (Tracer.current() != null) {
                Jaxpr subJaxpr = traceBody(fn, arg.shape(), arg.dtype());
                Tracer tracer = Tracer.current();
                Var inVar = arg instanceof TracedNDArray t ? t.getVar() : tracer.nextConstant(arg);
                Var outVar = tracer.nextVar(
                    subJaxpr.outVars().get(0).shape(),
                    subJaxpr.outVars().get(0).dtype());
                tracer.addEquation(new Equation(
                    List.of(inVar), List.of(outVar),
                    Primitive.CHECKPOINT, new CheckpointMeta(subJaxpr)));
                return new TracedNDArray(outVar);
            }
            return fn.apply(arg);
        };
    }

    /**
     * Returns a function that computes the gradient of fn.
     */
    public static Function<NDArray, NDArray> grad(Function<NDArray, NDArray> fn) {
        return Grad.grad(fn);
    }

    /**
     * Returns a function that computes the gradient of a two-argument fn with
     * respect to the argument selected by {@code argnum} (0 or 1).
     */
    public static BiFunction<NDArray, NDArray, NDArray> grad(BiFunction<NDArray, NDArray, NDArray> fn, int argnum) {
        return Grad.grad(fn, argnum);
    }

    /**
     * Returns a function that computes the gradients of a two-argument fn with
     * respect to both arguments at once.
     */
    public static BiFunction<NDArray, NDArray, NDArray[]> gradBoth(BiFunction<NDArray, NDArray, NDArray> fn) {
        return Grad.gradBoth(fn);
    }

    /**
     * Returns a function that computes the gradient of fn with respect to every
     * leaf of a {@link PyTree} of parameters, packaged back into the same structure.
     */
    public static Function<PyTree, PyTree> gradTree(Function<PyTree, NDArray> fn) {
        return Grad.gradTree(fn);
    }

    /**
     * Vectorizes a function over its leading (axis-0) batch dimension, mirroring
     * {@code jax.vmap} with the default {@code in_axes=0, out_axes=0}.
     */
    public static Function<NDArray, NDArray> vmap(Function<NDArray, NDArray> fn) {
        return Vmap.vmap(fn);
    }

    /**
     * Two-argument {@code vmap}: both arguments are batched along axis 0 and
     * the output batch dimension is at axis 0.  Mirrors
     * {@code jax.vmap(fn)} where {@code fn} takes two arrays and
     * {@code in_axes=0, out_axes=0}.
     */
    public static BiFunction<NDArray, NDArray, NDArray> vmap(BiFunction<NDArray, NDArray, NDArray> fn) {
        return Vmap.vmap(fn);
    }

    /**
     * Two-argument {@code vmap} with explicit axis control, mirroring
     * {@code jax.vmap(fn, in_axes=..., out_axes=...)}.
     *
     * <p>{@code inAxes[i]} is the axis carrying the batch dimension for the
     * i-th argument ({@code -1} = not batched, equivalent to Python's
     * {@code None}).  {@code outAxis} is the axis where the batch dimension
     * appears in the output.
     */
    public static BiFunction<NDArray, NDArray, NDArray> vmap(
            BiFunction<NDArray, NDArray, NDArray> fn, int[] inAxes, int outAxis) {
        return Vmap.vmap(fn, inAxes, outAxis);
    }

    /**
     * Parallelizes a function across a set of devices, mirroring
     * {@code jax.pmap} with {@code in_axes=0, out_axes=0}. The input's leading
     * dimension must equal {@code devices.size()}; each device receives the
     * corresponding shard (one slice along axis 0). Results are stacked back
     * into shape {@code [numDevices, *per-device-output-shape]}.
     *
     * <p>When the result of {@code pmap(f)} is used inside {@code JAX.grad},
     * a {@code PMAP} equation is emitted into the surrounding Jaxpr and its
     * VJP runs the backward pass on each shard concurrently — so gradients
     * compose correctly with collectives ({@link Lax#psum},
     * {@link Lax#allGather}) used inside the body function.
     *
     * <p>On a machine without real accelerators, pass two (or more) synthetic
     * {@code new Device(null)} instances to test the sharding and collective
     * logic on the CPU.
     */
    public static Function<NDArray, NDArray> pmap(Function<NDArray, NDArray> fn, List<Device> devices) {
        int D = devices.size();
        return (input) -> {
            if (Tracer.current() != null) {
                // Being traced (e.g. inside JAX.grad): emit a PMAP equation.
                int[] dims = input.shape().dimensions();
                com.marmanis.jax4j.core.Shape shardShape =
                    new com.marmanis.jax4j.core.Shape(Arrays.copyOfRange(dims, 1, dims.length));
                Jaxpr bodyJaxpr = traceBody(fn, shardShape, input.dtype());
                Tracer tracer = Tracer.current();
                Var inVar = input instanceof TracedNDArray t ? t.getVar()
                                                              : tracer.nextConstant(input);
                com.marmanis.jax4j.core.Shape outShape =
                    ScanUtil.prependDim(bodyJaxpr.outVars().get(0).shape(), D);
                Var outVar = tracer.nextVar(outShape, input.dtype());
                tracer.addEquation(new Equation(List.of(inVar), List.of(outVar),
                    Primitive.PMAP, new PmapMeta(bodyJaxpr, devices)));
                return new TracedNDArray(outVar);
            }
            return Pmap.runForward(fn, input, devices);
        };
    }

    private static Jaxpr traceBody(Function<NDArray, NDArray> fn,
                                    com.marmanis.jax4j.core.Shape shardShape, DType dtype) {
        Tracer.start();
        try {
            Var inVar = Tracer.current().nextVar(shardShape, dtype);
            NDArray result = fn.apply(new TracedNDArray(inVar));
            Var outVar = ((TracedNDArray) result).getVar();
            return Tracer.stop(List.of(inVar), List.of(outVar));
        } catch (RuntimeException | Error e) {
            Tracer.abort();
            throw e;
        }
    }
}
