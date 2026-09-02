package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SliceTest {

    @Test
    public void sliceForwardStep1() {
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7,8,9,10,11}, new Shape(3, 4));
        NDArray r = x.slice(new int[]{0, 1}, new int[]{2, 3});
        assertEquals(new Shape(2, 2), r.shape());
        assertArrayEquals(new float[]{1,2,5,6}, r.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceStopMinusOne() {
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7,8,9,10,11}, new Shape(3, 4));
        NDArray r = x.slice(new int[]{1, 0}, new int[]{-1, -1});
        assertEquals(new Shape(2, 4), r.shape());
        assertArrayEquals(new float[]{4,5,6,7,8,9,10,11}, r.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceStep2() {
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7}, new Shape(8));
        NDArray r = x.slice(new int[]{0}, new int[]{8}, new int[]{2});
        assertEquals(new Shape(4), r.shape());
        assertArrayEquals(new float[]{0,2,4,6}, r.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceAxisShortcut() {
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7,8,9,10,11}, new Shape(3, 4));
        NDArray r = x.sliceAxis(1, 1, 3);
        assertEquals(new Shape(3, 2), r.shape());
        assertArrayEquals(new float[]{1,2,5,6,9,10}, r.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceIndexReducesRank() {
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7,8,9,10,11}, new Shape(3, 4));
        NDArray r = x.sliceIndex(0, 2);
        assertEquals(new Shape(4), r.shape());
        assertArrayEquals(new float[]{8,9,10,11}, r.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceGradientStep1() {
        // f(x) = sum(x[1:3, 1:3])
        Function<NDArray, NDArray> fn = x -> x.slice(new int[]{1, 1}, new int[]{3, 3}).sum();
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, new Shape(4, 4));
        NDArray g = JAX.grad(fn).apply(x);
        // Ones at [1:3, 1:3]
        float[] exp = new float[16];
        for (int i = 1; i < 3; i++) for (int j = 1; j < 3; j++) exp[i * 4 + j] = 1f;
        assertArrayEquals(exp, g.toFloatArray(), 1e-6f);
    }

    @Test
    public void sliceGradientStep2() {
        // f(x) = sum(x[::2]) on shape (6,)
        Function<NDArray, NDArray> fn = x -> x.slice(new int[]{0}, new int[]{6}, new int[]{2}).sum();
        NDArray x = new ConcreteNDArray(new float[]{0,1,2,3,4,5}, new Shape(6));
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new float[]{1,0,1,0,1,0}, g.toFloatArray(), 1e-6f);
    }
}
