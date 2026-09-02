package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MaxMinAxisTest {

    // (2, 3) matrix: rows [1,2,3] and [10,20,30]
    private static NDArray matrix() {
        return new ConcreteNDArray(new float[]{1, 2, 3, 10, 20, 30}, new Shape(2, 3));
    }

    @Test
    public void maxAxisLast() {
        NDArray r = matrix().max(1);
        assertEquals(new Shape(2), r.shape());
        assertArrayEquals(new float[]{3f, 30f}, r.toFloatArray(), 1e-5f);
    }

    @Test
    public void maxAxisKeepDims() {
        NDArray r = matrix().max(1, true);
        assertEquals(new Shape(2, 1), r.shape());
        assertArrayEquals(new float[]{3f, 30f}, r.toFloatArray(), 1e-5f);
    }

    @Test
    public void minAxisFirst() {
        NDArray r = matrix().min(0);
        assertEquals(new Shape(3), r.shape());
        assertArrayEquals(new float[]{1f, 2f, 3f}, r.toFloatArray(), 1e-5f);
    }

    @Test
    public void maxAxisGradientRoutes() {
        // f(x) = sum(max(x, axis=1)). Gradient is 1 at argmax positions.
        Function<NDArray, NDArray> fn = x -> x.max(1).sum();
        NDArray x = matrix();
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new float[]{0, 0, 1, 0, 0, 1}, g.toFloatArray(), 1e-6f);
    }

    @Test
    public void minAxisGradientRoutes() {
        Function<NDArray, NDArray> fn = x -> x.min(1).sum();
        NDArray g = JAX.grad(fn).apply(matrix());
        assertArrayEquals(new float[]{1, 0, 0, 1, 0, 0}, g.toFloatArray(), 1e-6f);
    }

    @Test
    public void maxAxisTiesSplitEqually() {
        // Row 0 has two ties at max
        NDArray x = new ConcreteNDArray(new float[]{3, 3, 1, 10, 20, 30}, new Shape(2, 3));
        Function<NDArray, NDArray> fn = a -> a.max(1).sum();
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new float[]{0.5f, 0.5f, 0, 0, 0, 1}, g.toFloatArray(), 1e-6f);
    }

    @Test
    public void numericalGradMax() {
        Function<NDArray, NDArray> fn = a -> a.max(0).sum();
        // Distinct values -> no tie issues.
        NDArray x = new ConcreteNDArray(new float[]{1.2f, 5.7f, 3.3f, 9.1f, 2.8f, 4.4f}, new Shape(2, 3));
        NDArray g = JAX.grad(fn).apply(x);
        float h = 1e-3f;
        float[] xd = x.toFloatArray().clone();
        for (int i = 0; i < xd.length; i++) {
            float[] xp = xd.clone(); xp[i] += h;
            float[] xm = xd.clone(); xm[i] -= h;
            float fp = fn.apply(new ConcreteNDArray(xp, x.shape())).toFloatArray()[0];
            float fm = fn.apply(new ConcreteNDArray(xm, x.shape())).toFloatArray()[0];
            assertEquals((fp - fm) / (2 * h), g.toFloatArray()[i], 1e-2f);
        }
    }
}
