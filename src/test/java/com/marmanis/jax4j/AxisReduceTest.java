package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@code NDArray.sum(axis, keepDims)}/{@code mean(axis, keepDims)},
 * mirroring {@code numpy.sum(axis=, keepdims=)} semantics: forward values,
 * autodiff via {@code grad}, and composition with {@code vmap}.
 */
public class AxisReduceTest {

    // (2, 3) matrix: rows [1,2,3] and [10,20,30]
    private static NDArray matrix() {
        return new ConcreteNDArray(new float[]{1, 2, 3, 10, 20, 30}, new Shape(2, 3));
    }

    @Test
    public void sumAxisLastDropsRank() {
        NDArray result = matrix().sum(1);
        assertEquals(new Shape(2), result.shape());
        assertArrayEquals(new float[]{6f, 60f}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void sumAxisKeepDimsPreservesRank() {
        NDArray result = matrix().sum(1, true);
        assertEquals(new Shape(2, 1), result.shape());
        assertArrayEquals(new float[]{6f, 60f}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void sumAxisNegativeMatchesLastAxis() {
        NDArray result = matrix().sum(-1, true);
        assertEquals(new Shape(2, 1), result.shape());
        assertArrayEquals(new float[]{6f, 60f}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void sumAxisFirstDimension() {
        // column sums: [1+10, 2+20, 3+30]
        NDArray result = matrix().sum(0);
        assertEquals(new Shape(3), result.shape());
        assertArrayEquals(new float[]{11f, 22f, 33f}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void meanAxisMatchesExpected() {
        NDArray result = matrix().mean(1, true);
        assertEquals(new Shape(2, 1), result.shape());
        assertArrayEquals(new float[]{2f, 20f}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void sumAxisBroadcastsBackForDivision() {
        // x / sum(x, axis=-1, keepdims=True) -- each row normalized to sum to 1
        NDArray x = matrix();
        NDArray normalized = x.div(x.sum(-1, true));
        float[] result = normalized.toFloatArray();
        assertEquals(1f, result[0] + result[1] + result[2], 1e-5f);
        assertEquals(1f, result[3] + result[4] + result[5], 1e-5f);
    }

    @Test
    public void sumAxisGradientBroadcastsBackToEveryElement() {
        // f(x) = sum(sum(x, axis=1)) ; d/dx_ij = 1 for every element
        Function<NDArray, NDArray> fn = x -> x.sum(1).sum();
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        float[] grad = gradFn.apply(matrix()).toFloatArray();
        assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1}, grad, 1e-5f);
    }

    @Test
    public void meanAxisGradientScalesByAxisSize() {
        // f(x) = sum(mean(x, axis=1)) ; d/dx_ij = 1/axisSize = 1/3
        Function<NDArray, NDArray> fn = x -> x.mean(1).sum();
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        float[] grad = gradFn.apply(matrix()).toFloatArray();
        float expected = 1f / 3f;
        assertArrayEquals(new float[]{expected, expected, expected, expected, expected, expected}, grad, 1e-5f);
    }

    @Test
    public void sumAxisKeepDimsGradientMatchesNonKeepDims() {
        // Same reduction, just keepDims=true; gradient should be identical.
        Function<NDArray, NDArray> fn = x -> x.sum(1, true).sum();
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        float[] grad = gradFn.apply(matrix()).toFloatArray();
        assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1}, grad, 1e-5f);
    }

    @Test
    public void vmapOfAxisSumShiftsAxisForBatchDimension() {
        // Per-example function reduces a (3, 2) matrix along axis=1 (row sums).
        // vmap should batch this correctly over a (batch, 3, 2) input.
        Function<NDArray, NDArray> fn = x -> x.sum(1);
        NDArray batch = new ConcreteNDArray(
            new float[]{
                1, 2, 3, 4, 5, 6,       // example 0: rows sum to [3, 7, 11]
                10, 20, 30, 40, 50, 60  // example 1: rows sum to [30, 70, 110]
            },
            new Shape(2, 3, 2));

        NDArray batched = JAX.vmap(fn).apply(batch);

        assertEquals(new Shape(2, 3), batched.shape());
        assertArrayEquals(new float[]{3, 7, 11, 30, 70, 110}, batched.toFloatArray(), 1e-5f);
    }
}
