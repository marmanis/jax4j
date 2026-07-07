package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.CondMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.ScanMeta;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.ir.WhileMeta;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static com.marmanis.jax4j.api.ScanUtil.dropLeadingDim;
import static com.marmanis.jax4j.api.ScanUtil.prependDim;
import static com.marmanis.jax4j.api.ScanUtil.sliceLeading;
import static com.marmanis.jax4j.api.ScanUtil.stackLeading;

/**
 * Control-flow primitives mirroring {@code jax.lax}. Unlike a plain Java
 * {@code if}/{@code for} wrapped around traced {@link NDArray} operations,
 * {@link #cond}, {@link #whileLoop} and {@link #scan} record themselves as a
 * single equation in the surrounding {@link Jaxpr} — with the branch/
 * condition/body/step functions traced into their own nested Jaxprs — so
 * they survive {@code grad}/{@code gradTree} as a real part of the
 * computation graph instead of being unrolled away outside it.
 *
 * <p>{@link #foriLoop} is the exception: its bound is a plain Java
 * {@code int} known at trace time, so it's simply a Java loop calling
 * {@code bodyFn} repeatedly — each call appends its ops directly to whatever
 * Jaxpr (if any) is currently being built, which is already fully traceable
 * and differentiable with no new primitive needed.
 *
 * <p>Outside any active trace (e.g. called at the top level on a plain
 * {@link com.marmanis.jax4j.core.ConcreteNDArray}), {@code cond}/{@code
 * whileLoop}/{@code scan} all execute eagerly with no Jaxpr involved at all —
 * exactly like {@code NDArray.add} runs a plain Java loop outside a trace
 * instead of recording an equation.
 *
 * <p>{@code whileLoop} has no VJP rule and is not reverse-mode
 * differentiable, mirroring {@code jax.lax.while_loop}: its iteration count
 * is itself data-dependent, so there is no fixed-size backward pass to run.
 * Use {@code scan} or {@code foriLoop} (both have a trace-time-known number
 * of steps) when gradients through repeated application are needed.
 */
public final class Lax {
    private Lax() {}

    private static boolean isTracing() {
        return Tracer.current() != null;
    }

    private static boolean isTrue(NDArray pred) {
        return pred.toFloatArray()[0] != 0f;
    }

    private static Var toVar(NDArray a) {
        if (a instanceof TracedNDArray t) return t.getVar();
        return Tracer.current().nextConstant(a);
    }

    // ---- cond ----

    /** Mirrors {@code jax.lax.cond(pred, true_fn, false_fn, operand)}. */
    public static NDArray cond(NDArray pred, Function<NDArray, NDArray> trueFn, Function<NDArray, NDArray> falseFn, NDArray operand) {
        if (!isTracing()) {
            return isTrue(pred) ? trueFn.apply(operand) : falseFn.apply(operand);
        }

        Jaxpr trueBranch = JAX.make_jaxpr(trueFn, operand);
        Jaxpr falseBranch = JAX.make_jaxpr(falseFn, operand);

        Tracer tracer = Tracer.current();
        Var predVar = toVar(pred);
        Var operandVar = toVar(operand);
        Var outVar = tracer.nextVar(trueBranch.outVars().get(0).shape(), operand.dtype());
        tracer.addEquation(new Equation(List.of(predVar, operandVar), List.of(outVar), Primitive.COND,
            new CondMeta(trueBranch, falseBranch)));
        return new TracedNDArray(outVar);
    }

    // ---- while_loop ----

    /** Mirrors {@code jax.lax.while_loop(cond_fun, body_fun, init_val)}. Not reverse-mode differentiable. */
    public static NDArray whileLoop(Function<NDArray, NDArray> condFn, Function<NDArray, NDArray> bodyFn, NDArray init) {
        if (!isTracing()) {
            NDArray state = init;
            while (isTrue(condFn.apply(state))) {
                state = bodyFn.apply(state);
            }
            return state;
        }

        Jaxpr condJaxpr = JAX.make_jaxpr(condFn, init);
        Jaxpr bodyJaxpr = JAX.make_jaxpr(bodyFn, init);

        Tracer tracer = Tracer.current();
        Var initVar = toVar(init);
        Var outVar = tracer.nextVar(bodyJaxpr.outVars().get(0).shape(), init.dtype());
        tracer.addEquation(new Equation(List.of(initVar), List.of(outVar), Primitive.WHILE,
            new WhileMeta(condJaxpr, bodyJaxpr)));
        return new TracedNDArray(outVar);
    }

    // ---- fori_loop ----

    /**
     * Mirrors {@code jax.lax.fori_loop(lower, upper, body_fun, init_val)}.
     * {@code lower}/{@code upper} are plain Java {@code int}s, so this is
     * just {@code upper - lower} calls to {@code bodyFn}.
     */
    public static NDArray foriLoop(int lower, int upper, BiFunction<Integer, NDArray, NDArray> bodyFn, NDArray init) {
        NDArray state = init;
        for (int i = lower; i < upper; i++) {
            state = bodyFn.apply(i, state);
        }
        return state;
    }

    // ---- scan ----

    public record ScanResult(NDArray carry, NDArray ys) {}

    /**
     * Mirrors {@code jax.lax.scan(f, init, xs)}: applies {@code stepFn} to
     * {@code (carry, xs[t])} for each leading-axis slice of {@code xs} in
     * order, threading the carry through and stacking each step's second
     * output into {@code ys}. {@code stepFn} returns {@code [newCarry, y]}.
     */
    public static ScanResult scan(BiFunction<NDArray, NDArray, NDArray[]> stepFn, NDArray initCarry, NDArray xs) {
        int steps = xs.shape().dimensions()[0];
        Shape xStepShape = dropLeadingDim(xs.shape());

        if (!isTracing()) {
            NDArray carry = initCarry;
            List<NDArray> ys = new ArrayList<>(steps);
            for (int t = 0; t < steps; t++) {
                NDArray[] out = stepFn.apply(carry, sliceLeading(xs, t, xStepShape));
                carry = out[0];
                ys.add(out[1]);
            }
            return new ScanResult(carry, stackLeading(ys));
        }

        Jaxpr stepJaxpr = traceStep(stepFn, initCarry.shape(), initCarry.dtype(), xStepShape, xs.dtype());

        Tracer tracer = Tracer.current();
        Var carryVar = toVar(initCarry);
        Var xsVar = toVar(xs);
        Shape yStepShape = stepJaxpr.outVars().get(1).shape();
        Var carryOutVar = tracer.nextVar(stepJaxpr.outVars().get(0).shape(), initCarry.dtype());
        Var ysOutVar = tracer.nextVar(prependDim(yStepShape, steps), xs.dtype());
        tracer.addEquation(new Equation(List.of(carryVar, xsVar), List.of(carryOutVar, ysOutVar), Primitive.SCAN,
            new ScanMeta(stepJaxpr)));
        return new ScanResult(new TracedNDArray(carryOutVar), new TracedNDArray(ysOutVar));
    }

    // ---- collectives (for use inside pmap bodies) ----

    /**
     * All-reduce sum across all pmap shards: each shard's value is summed and
     * the total is returned to all shards. Outside a pmap context this is the
     * identity, mirroring {@code jax.lax.psum} with {@code axis_name}.
     *
     * <p>Inside a traced pmap body this records a {@code PSUM} equation so
     * that {@code grad(pmap(f))} correctly back-propagates through the collective.
     */
    public static NDArray psum(NDArray x) {
        if (isTracing()) {
            Tracer tracer = Tracer.current();
            Var xVar = toVar(x);
            Var outVar = tracer.nextVar(x.shape(), x.dtype());
            tracer.addEquation(new Equation(List.of(xVar), List.of(outVar), Primitive.PSUM, null));
            return new TracedNDArray(outVar);
        }
        PmapContext ctx = PmapContext.current();
        if (ctx == null) return x;
        return new ConcreteNDArray(ctx.collective.psum(ctx.deviceIndex, x.toFloatArray()), x.shape());
    }

    /**
     * All-gather: each shard contributes its local value and all shards receive
     * the stacked result of shape {@code [numDevices, *x.shape()]}. Outside a
     * pmap context this prepends a size-1 leading dimension.
     *
     * <p>{@code numDevices} must match the enclosing {@code pmap}'s device
     * count; it is required here so that the output shape is known at trace time
     * (mirroring JAX's requirement that the axis size is statically known).
     *
     * <p>Inside a traced body this records an {@code ALL_GATHER} equation
     * so that {@code grad(pmap(f))} correctly back-propagates the slice
     * gradient to each device.
     */
    public static NDArray allGather(NDArray x, int numDevices) {
        if (isTracing()) {
            Tracer tracer = Tracer.current();
            Var xVar = toVar(x);
            Shape outShape = ScanUtil.prependDim(x.shape(), numDevices);
            Var outVar = tracer.nextVar(outShape, x.dtype());
            tracer.addEquation(new Equation(List.of(xVar), List.of(outVar), Primitive.ALL_GATHER, numDevices));
            return new TracedNDArray(outVar);
        }
        PmapContext ctx = PmapContext.current();
        if (ctx == null) {
            // Outside pmap: wrap in a [1, *] array
            return ScanUtil.stackLeading(List.of(x));
        }
        return ctx.collective.allGather(ctx.deviceIndex, x);
    }

    private static Jaxpr traceStep(BiFunction<NDArray, NDArray, NDArray[]> stepFn,
                                    Shape carryShape, DType carryDtype, Shape xShape, DType xDtype) {
        Tracer.start();
        try {
            Var carryVar = Tracer.current().nextVar(carryShape, carryDtype);
            Var xVar = Tracer.current().nextVar(xShape, xDtype);
            NDArray[] outs = stepFn.apply(new TracedNDArray(carryVar), new TracedNDArray(xVar));
            Var newCarryVar = ((TracedNDArray) outs[0]).getVar();
            Var yVar = ((TracedNDArray) outs[1]).getVar();
            return Tracer.stop(List.of(carryVar, xVar), List.of(newCarryVar, yVar));
        } catch (RuntimeException | Error e) {
            Tracer.abort();
            throw e;
        }
    }
}
