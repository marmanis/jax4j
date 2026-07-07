package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.testutil.GradChecker;
import org.junit.jupiter.api.Test;

import java.util.function.BiFunction;
import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.assertGradMatches;
import static com.marmanis.jax4j.testutil.GradChecker.vector;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Gradient-correctness tests for each jax4j primitive, modeled on the role of
 * upstream JAX's {@code tests/lax_autodiff_test.py}: every primitive's VJP is
 * checked against a numerical (finite-difference) gradient rather than only
 * against one hand-picked closed-form value.
 */
public class LaxAutodiffTest {

    private static NDArray grad1(Function<NDArray, NDArray> fn, NDArray x) {
        return JAX.grad(fn).apply(x);
    }

    @Test
    public void testAddGrad() {
        Function<NDArray, NDArray> fn = x -> x.add(vector(1, 2, 3)).sum();
        NDArray x = vector(0.5f, -1.2f, 3.4f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testSubGrad() {
        Function<NDArray, NDArray> fn = x -> x.sub(vector(1, 2, 3)).sum();
        NDArray x = vector(0.5f, -1.2f, 3.4f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testMulGrad() {
        Function<NDArray, NDArray> fn = x -> x.mul(vector(2, -3, 4)).sum();
        NDArray x = vector(0.5f, -1.2f, 3.4f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testDivGrad() {
        Function<NDArray, NDArray> fn = x -> x.div(vector(2, -3, 4)).sum();
        NDArray x = vector(0.5f, -1.2f, 3.4f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testExpGrad() {
        Function<NDArray, NDArray> fn = x -> x.exp().sum();
        NDArray x = vector(0.1f, -0.5f, 1.3f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testLogGrad() {
        Function<NDArray, NDArray> fn = x -> x.log().sum();
        NDArray x = vector(0.5f, 1.5f, 3.0f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testSinGrad() {
        Function<NDArray, NDArray> fn = x -> x.sin().sum();
        NDArray x = vector(0.2f, -0.8f, 2.1f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testCosGrad() {
        Function<NDArray, NDArray> fn = x -> x.cos().sum();
        NDArray x = vector(0.2f, -0.8f, 2.1f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testSumGrad() {
        Function<NDArray, NDArray> fn = NDArray::sum;
        NDArray x = vector(1, 2, 3, 4);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testMeanGrad() {
        Function<NDArray, NDArray> fn = NDArray::mean;
        NDArray x = vector(1, 2, 3, 4);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testComposedGrad() {
        // f(x) = sum(sin(x * x) + log(x))  -- chain of several primitives at once
        Function<NDArray, NDArray> fn = x -> x.mul(x).sin().add(x.log()).sum();
        NDArray x = vector(0.6f, 1.1f, 2.2f);
        assertGradMatches(fn, grad1(fn, x), x);
    }

    @Test
    public void testDotGradAgainstNumeric() {
        NDArray b = new ConcreteNDArray(new float[]{1.5f, -2.0f, 0.5f, 3.0f}, new Shape(2, 2));
        Function<NDArray, NDArray> fn = a -> a.dot(b).sum();
        NDArray a = new ConcreteNDArray(new float[]{0.3f, -0.7f, 1.1f, 2.2f}, new Shape(2, 2));
        assertGradMatches(fn, grad1(fn, a), a);
    }

    @Test
    public void testGradOfGrad() {
        // f(x) = x^3 ; f'(x) = 3x^2 ; f''(x) = 6x
        Function<NDArray, NDArray> f = x -> x.mul(x).mul(x);
        Function<NDArray, NDArray> df = JAX.grad(f);
        Function<NDArray, NDArray> d2f = JAX.grad(df);

        NDArray x = vector(2.0f);
        assertEquals(6.0f * 2.0f, d2f.apply(x).toFloatArray()[0], 1e-3f);
    }

    @Test
    public void testGradWithRespectToEachArgnum() {
        // f(a, b) = a * b + b ; df/da = b ; df/db = a + 1
        BiFunction<NDArray, NDArray, NDArray> f = (a, b) -> a.mul(b).add(b);
        NDArray a = vector(3.0f);
        NDArray b = vector(5.0f);

        NDArray dfda = JAX.grad(f, 0).apply(a, b);
        NDArray dfdb = JAX.grad(f, 1).apply(a, b);

        assertEquals(5.0f, dfda.toFloatArray()[0], 1e-5f);
        assertEquals(4.0f, dfdb.toFloatArray()[0], 1e-5f);
    }

    @Test
    public void testGradBothArgnumsAtOnce() {
        BiFunction<NDArray, NDArray, NDArray> f = (a, b) -> a.mul(b).add(b);
        NDArray a = vector(3.0f);
        NDArray b = vector(5.0f);

        NDArray[] grads = JAX.gradBoth(f).apply(a, b);

        assertEquals(5.0f, grads[0].toFloatArray()[0], 1e-5f);
        assertEquals(4.0f, grads[1].toFloatArray()[0], 1e-5f);
    }
}
