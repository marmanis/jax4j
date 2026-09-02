package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SqrtRsqrtTest {

    @Test
    public void sqrtForward() {
        NDArray x = new ConcreteNDArray(new float[]{1f, 4f, 9f, 16f}, new Shape(4));
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, x.sqrt().toFloatArray(), 1e-5f);
    }

    @Test
    public void rsqrtForward() {
        NDArray x = new ConcreteNDArray(new float[]{1f, 4f, 16f}, new Shape(3));
        assertArrayEquals(new float[]{1f, 0.5f, 0.25f}, x.rsqrt().toFloatArray(), 1e-5f);
    }

    @Test
    public void sqrtGradient() {
        // f(x) = sum(sqrt(x)); df/dx_i = 0.5/sqrt(x_i)
        Function<NDArray, NDArray> fn = x -> x.sqrt().sum();
        NDArray x = new ConcreteNDArray(new float[]{1f, 4f, 9f, 16f}, new Shape(4));
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new float[]{0.5f, 0.25f, 1f/6f, 0.125f}, g.toFloatArray(), 1e-4f);
    }

    @Test
    public void rsqrtGradient() {
        // f(x) = sum(rsqrt(x)); df/dx_i = -0.5 * x_i^(-3/2)
        Function<NDArray, NDArray> fn = x -> x.rsqrt().sum();
        NDArray x = new ConcreteNDArray(new float[]{1f, 4f, 16f}, new Shape(3));
        NDArray g = JAX.grad(fn).apply(x);
        float[] r = g.toFloatArray();
        assertEquals(-0.5f, r[0], 1e-4f);
        assertEquals(-0.5f * (float) Math.pow(4.0, -1.5), r[1], 1e-4f);
        assertEquals(-0.5f * (float) Math.pow(16.0, -1.5), r[2], 1e-5f);
    }

    @Test
    public void rsqrtHandlesSmallPositiveValues() {
        NDArray x = new ConcreteNDArray(new float[]{1e-6f}, new Shape(1));
        float v = x.rsqrt().toFloatArray()[0];
        assertEquals(1e3f, v, 1f);
    }

    @Test
    public void numericalGradSqrt() {
        Function<NDArray, NDArray> fn = x -> x.sqrt().sum();
        NDArray x = new ConcreteNDArray(new float[]{2.5f, 6.25f}, new Shape(2));
        NDArray g = JAX.grad(fn).apply(x);
        // Numerical
        float h = 1e-3f;
        for (int i = 0; i < 2; i++) {
            float[] xp = x.toFloatArray().clone(); xp[i] += h;
            float[] xm = x.toFloatArray().clone(); xm[i] -= h;
            float fp = fn.apply(new ConcreteNDArray(xp, x.shape())).toFloatArray()[0];
            float fm = fn.apply(new ConcreteNDArray(xm, x.shape())).toFloatArray()[0];
            float numeric = (fp - fm) / (2 * h);
            assertEquals(numeric, g.toFloatArray()[i], 1e-2f);
        }
    }
}
