package com.marmanis.jax4j.tracing;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.Var;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Manages the tracing state for one thread.
 *
 * <p>Tracers are kept on a per-thread stack so that nested traces compose
 * correctly ({@code grad(grad(f))}) and concurrent pmap shards running on
 * separate threads cannot corrupt each other's trace state.
 *
 * <p>The per-thread stack is the right abstraction here: {@code start()}/
 * {@code stop()} are explicit push/pop operations whose lifetimes don't align
 * with Java stack frames, so they cannot use try-with-resources or the
 * callback-based {@code ScopedValue.where(...).run(...)} API directly.
 * The {@code ThreadLocal} gives us the same thread-confinement guarantee
 * that {@code ScopedValue} provides, without requiring callers to restructure
 * around a callback boundary.
 */
public class Tracer {
    private static final ThreadLocal<Deque<Tracer>> stackHolder =
        ThreadLocal.withInitial(ArrayDeque::new);

    private final List<Equation> equations = new ArrayList<>();
    private final AtomicInteger varCounter = new AtomicInteger(0);
    private final Map<Integer, NDArray> consts = new HashMap<>();

    public static void start() {
        stackHolder.get().push(new Tracer());
    }

    public static Tracer current() {
        return stackHolder.get().peek();
    }

    public static Jaxpr stop(List<Var> inVars, List<Var> outVars) {
        Tracer tracer = stackHolder.get().pop();
        return new Jaxpr(inVars, outVars, tracer.equations, tracer.consts);
    }

    /**
     * Pops the current tracer without building a {@link Jaxpr}. Callers that
     * wrap {@link #start()}/{@link #stop} around a user-supplied function
     * (e.g. {@code make_jaxpr}, {@code gradTree}, {@code Lax.scan}'s step
     * tracing) must call this in a {@code catch} block if that function
     * throws — otherwise the tracer it pushed is never popped, and every
     * later call on this thread sees a stale {@link #current()}, silently
     * treating plain eager code as if it were being traced.
     */
    public static void abort() {
        stackHolder.get().pop();
    }

    public Var nextVar(com.marmanis.jax4j.core.Shape shape, com.marmanis.jax4j.core.DType dtype) {
        return new Var(varCounter.getAndIncrement(), shape, dtype);
    }

    public Var nextConstant(NDArray value) {
        Var v = nextVar(value.shape(), value.dtype());
        consts.put(v.id(), value);
        return v;
    }

    public void addEquation(Equation eq) {
        equations.add(eq);
    }
}
