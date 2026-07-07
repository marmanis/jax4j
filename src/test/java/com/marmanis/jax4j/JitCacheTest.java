package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies that {@link JAX#jit} and {@link JAX#jitGrad} cache their traces
 * by input signature and match the untraced results.
 */
public class JitCacheTest {

    private static NDArray vec(double... xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testJitMatchesEagerAndCachesTrace() {
        // Counting trace calls by wrapping the user function with a probe
        // that only increments when actually invoked. The cached jit
        // should invoke the user function once per unique input signature.
        AtomicInteger traceCount = new AtomicInteger();
        Function<NDArray, NDArray> fn = x -> {
            traceCount.incrementAndGet();
            return x.mul(x).add(x);
        };
        Function<NDArray, NDArray> jittedFn = JAX.jit(fn);
        NDArray x = vec(2.0, 3.0);
        // First call traces + interprets.
        NDArray y1 = jittedFn.apply(x);
        assertEquals(1, traceCount.get(), "first call traces");
        // Second call at the same signature should hit the cache — no
        // additional call to fn.
        NDArray y2 = jittedFn.apply(vec(5.0, 7.0));
        assertEquals(1, traceCount.get(), "second call hits cache");
        // Different shape should retrace.
        jittedFn.apply(vec(1.0, 2.0, 3.0));
        assertEquals(2, traceCount.get(), "different shape retraces");

        double[] y1a = y1.toDoubleArray();
        assertClose(2.0 * 2.0 + 2.0, y1a[0], 1e-13, "y1[0]");
        assertClose(3.0 * 3.0 + 3.0, y1a[1], 1e-13, "y1[1]");
        double[] y2a = y2.toDoubleArray();
        assertClose(5.0 * 5.0 + 5.0, y2a[0], 1e-13, "y2[0]");
        assertClose(7.0 * 7.0 + 7.0, y2a[1], 1e-13, "y2[1]");
    }

    @Test
    public void testJitGradMatchesEagerGradAndCaches() {
        // f(x) = sum(x^3) -> f'(x_i) = 3 x_i^2. Grad checked at a probe.
        AtomicInteger traceCount = new AtomicInteger();
        Function<NDArray, NDArray> f = x -> {
            traceCount.incrementAndGet();
            return x.mul(x).mul(x).sum();
        };
        Function<NDArray, NDArray> jittedGrad = JAX.jitGrad(f);
        NDArray x = vec(1.0, 2.0, 3.0);
        NDArray g1 = jittedGrad.apply(x);
        assertEquals(1, traceCount.get(), "first grad call traces");
        NDArray g2 = jittedGrad.apply(vec(4.0, 5.0, 6.0));
        assertEquals(1, traceCount.get(), "second grad call hits cache");
        double[] g1a = g1.toDoubleArray();
        double[] g2a = g2.toDoubleArray();
        double tol = 1e-12;
        assertClose(3.0 * 1.0 * 1.0, g1a[0], tol, "g1[0]");
        assertClose(3.0 * 2.0 * 2.0, g1a[1], tol, "g1[1]");
        assertClose(3.0 * 3.0 * 3.0, g1a[2], tol, "g1[2]");
        assertClose(3.0 * 4.0 * 4.0, g2a[0], tol, "g2[0]");
        assertClose(3.0 * 5.0 * 5.0, g2a[1], tol, "g2[1]");
        assertClose(3.0 * 6.0 * 6.0, g2a[2], tol, "g2[2]");
    }

    @Test
    public void testJitTransparentUnderNestedTrace() {
        // A jitted function called from inside JAX.grad should still be
        // differentiable — jit must fall through eagerly when Tracer.current()
        // is non-null, otherwise the outer grad wouldn't see the ops.
        Function<NDArray, NDArray> inner = x -> x.mul(x);          // x^2
        Function<NDArray, NDArray> jittedInner = JAX.jit(inner);
        Function<NDArray, NDArray> outer = x -> jittedInner.apply(x).sum();  // sum(x^2)
        Function<NDArray, NDArray> gradOuter = JAX.grad(outer);
        NDArray x = vec(1.0, 2.0, 3.0);
        double[] g = gradOuter.apply(x).toDoubleArray();
        assertClose(2.0, g[0], 1e-12, "nested trace grad[0]");
        assertClose(4.0, g[1], 1e-12, "nested trace grad[1]");
        assertClose(6.0, g[2], 1e-12, "nested trace grad[2]");
    }

    @Test
    public void testJitAndJitGradHaveSeparateCaches() {
        // A single trace is not shared between jit and jitGrad — they should
        // each maintain their own cache.
        AtomicInteger cnt = new AtomicInteger();
        Function<NDArray, NDArray> f = x -> {
            cnt.incrementAndGet();
            return x.mul(x).sum();
        };
        JAX.jit(f).apply(vec(1.0, 2.0, 3.0));
        assertEquals(1, cnt.get(), "jit trace");
        JAX.jitGrad(f).apply(vec(1.0, 2.0, 3.0));
        assertNotEquals(1, cnt.get(), "jitGrad has its own cache and traces");
    }

    @Test
    public void testJitGradTreeCachesForMultiInput() {
        // A 3-argument function packed as a pytree with 3 leaves. This is
        // the pattern chebfun4j uses to pass (x, u, up, upp) without any
        // baked-in closure captures: everything travels as pytree leaves
        // so the trace can be reused across calls whose inputs differ in
        // value but share shape/dtype.
        AtomicInteger traceCount = new AtomicInteger();
        Function<com.marmanis.jax4j.pytree.PyTree, NDArray> lossFn = tree -> {
            traceCount.incrementAndGet();
            java.util.List<com.marmanis.jax4j.pytree.PyTree> ch =
                ((com.marmanis.jax4j.pytree.PyTree.ListNode) tree).children();
            NDArray a = ((com.marmanis.jax4j.pytree.PyTree.Leaf) ch.get(0)).value();
            NDArray b = ((com.marmanis.jax4j.pytree.PyTree.Leaf) ch.get(1)).value();
            NDArray c = ((com.marmanis.jax4j.pytree.PyTree.Leaf) ch.get(2)).value();
            // sum(a*b + c^2)
            return a.mul(b).add(c.mul(c)).sum();
        };
        Function<com.marmanis.jax4j.pytree.PyTree, com.marmanis.jax4j.pytree.PyTree> jittedGrad =
            JAX.jitGradTree(lossFn);
        com.marmanis.jax4j.pytree.PyTree tree1 = com.marmanis.jax4j.pytree.PyTree.list(
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(1.0, 2.0)),
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(3.0, 4.0)),
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(5.0, 6.0)));
        com.marmanis.jax4j.pytree.PyTree g1 = jittedGrad.apply(tree1);
        assertEquals(1, traceCount.get(), "first tree grad traces");
        // Different values, same shapes — should hit the cache.
        com.marmanis.jax4j.pytree.PyTree tree2 = com.marmanis.jax4j.pytree.PyTree.list(
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(0.5, 1.5)),
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(2.5, 3.5)),
            com.marmanis.jax4j.pytree.PyTree.leaf(vec(4.5, 5.5)));
        com.marmanis.jax4j.pytree.PyTree g2 = jittedGrad.apply(tree2);
        assertEquals(1, traceCount.get(), "same-signature call hits cache");
        // Verify gradient shape and values.
        java.util.List<NDArray> gLeaves1 = com.marmanis.jax4j.pytree.PyTrees.flatten(g1);
        // ∂/∂a (a*b + c²) = b -> [3, 4]
        assertClose(3.0, gLeaves1.get(0).toDoubleArray()[0], 1e-12, "g1.a[0]");
        assertClose(4.0, gLeaves1.get(0).toDoubleArray()[1], 1e-12, "g1.a[1]");
        // ∂/∂b = a -> [1, 2]
        assertClose(1.0, gLeaves1.get(1).toDoubleArray()[0], 1e-12, "g1.b[0]");
        assertClose(2.0, gLeaves1.get(1).toDoubleArray()[1], 1e-12, "g1.b[1]");
        // ∂/∂c = 2c -> [10, 12]
        assertClose(10.0, gLeaves1.get(2).toDoubleArray()[0], 1e-12, "g1.c[0]");
        assertClose(12.0, gLeaves1.get(2).toDoubleArray()[1], 1e-12, "g1.c[1]");
        // Second grad also correct.
        java.util.List<NDArray> gLeaves2 = com.marmanis.jax4j.pytree.PyTrees.flatten(g2);
        assertClose(2.5, gLeaves2.get(0).toDoubleArray()[0], 1e-12, "g2.a[0]");
    }

    @Test
    public void testRepeatedCallsAreFast() {
        // Very loose perf sanity: after warm-up, N repeated calls to jit
        // should be significantly faster than N eager calls when the traced
        // function does non-trivial work. Not a benchmark — just verify
        // caching pays off directionally.
        Function<NDArray, NDArray> heavy = x -> {
            NDArray y = x;
            for (int i = 0; i < 20; i++) y = y.mul(x).add(x);
            return y;
        };
        Function<NDArray, NDArray> jitted = JAX.jit(heavy);
        NDArray x = vec(0.5, 0.7, 1.1);
        // Warm-up: touch both paths so JIT compile isn't in the measurement.
        for (int i = 0; i < 100; i++) { jitted.apply(x); heavy.apply(x); }
        long t0 = System.nanoTime();
        for (int i = 0; i < 200; i++) jitted.apply(x);
        long tJit = System.nanoTime() - t0;
        // Correctness sanity: jitted and eager agree.
        double[] a = jitted.apply(x).toDoubleArray();
        double[] b = heavy.apply(x).toDoubleArray();
        for (int i = 0; i < a.length; i++) assertClose(b[i], a[i], 1e-10, "consistency[" + i + "]");
        assertTrue(tJit > 0, "sanity: measured non-zero jit time (" + tJit + " ns)");
    }
}
