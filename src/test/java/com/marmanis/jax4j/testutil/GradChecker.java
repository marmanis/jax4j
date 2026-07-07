package com.marmanis.jax4j.testutil;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.function.Function;

/**
 * Numerical-vs-analytic gradient checking, mirroring the role of
 * {@code jax.test_util.check_grads} in the upstream JAX test suite:
 * compares the gradient produced by {@code grad} against a central-difference
 * numerical approximation of the same function.
 */
public final class GradChecker {
    private GradChecker() {}

    public static final float DEFAULT_EPS = 1e-3f;
    public static final float DEFAULT_TOL = 5e-2f;

    /**
     * Computes df/dx at {@code x} via central differences, elementwise.
     */
    public static NDArray numericalGrad(Function<NDArray, NDArray> fn, NDArray x, float eps) {
        float[] xData = x.toFloatArray();
        float[] grad = new float[xData.length];
        for (int i = 0; i < xData.length; i++) {
            float[] plus = xData.clone();
            float[] minus = xData.clone();
            plus[i] += eps;
            minus[i] -= eps;
            float fPlus = sumOf(fn.apply(new ConcreteNDArray(plus, x.shape())));
            float fMinus = sumOf(fn.apply(new ConcreteNDArray(minus, x.shape())));
            grad[i] = (fPlus - fMinus) / (2 * eps);
        }
        return new ConcreteNDArray(grad, x.shape());
    }

    private static float sumOf(NDArray a) {
        float total = 0;
        for (float v : a.toFloatArray()) total += v;
        return total;
    }

    /**
     * Asserts that the analytic gradient (from {@code jax.grad}-style transforms)
     * matches the numerical gradient within {@code tol}, mirroring
     * {@code check_grads(fn, (x,), order=1)}.
     */
    public static void assertGradMatches(Function<NDArray, NDArray> fn, NDArray analyticGrad, NDArray x) {
        assertGradMatches(fn, analyticGrad, x, DEFAULT_EPS, DEFAULT_TOL);
    }

    public static void assertGradMatches(Function<NDArray, NDArray> fn, NDArray analyticGrad, NDArray x,
                                          float eps, float tol) {
        NDArray numeric = numericalGrad(fn, x, eps);
        float[] expected = numeric.toFloatArray();
        float[] actual = analyticGrad.toFloatArray();
        if (expected.length != actual.length) {
            throw new AssertionError("Gradient shape mismatch: numeric=" + numeric.shape()
                + " analytic=" + analyticGrad.shape());
        }
        for (int i = 0; i < expected.length; i++) {
            float diff = Math.abs(expected[i] - actual[i]);
            float scale = Math.max(1f, Math.abs(expected[i]));
            if (diff / scale > tol) {
                throw new AssertionError(String.format(
                    "Gradient mismatch at index %d: numeric=%.6f analytic=%.6f (tol=%.4f)",
                    i, expected[i], actual[i], tol));
            }
        }
    }

    public static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }

    public static NDArray vector(float... values) {
        return new ConcreteNDArray(values, new Shape(values.length));
    }
}
