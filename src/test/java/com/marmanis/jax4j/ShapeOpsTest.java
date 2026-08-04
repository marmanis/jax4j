package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Lax;
import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.core.DType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GAP-1 primitives: reshape, transpose, concat, pad, scatter_add.
 * Also covers autodiff (VJP) through these primitives.
 */
public class ShapeOpsTest {

    // ------------------------------------------------------------------
    // reshape
    // ------------------------------------------------------------------

    @Test
    public void testReshapeForwardShape() {
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray y = x.reshape(new Shape(3, 2));
        assertArrayEquals(new int[]{3, 2}, y.shape().dimensions());
        assertEquals(6, y.shape().size());
    }

    @Test
    public void testReshapePreservesData() {
        float[] data = {1, 2, 3, 4, 5, 6};
        NDArray x = new ConcreteNDArray(data, new Shape(6));
        NDArray y = x.reshape(new Shape(2, 3));
        float[] out = y.toFloatArray();
        assertArrayEquals(data, out, 1e-6f);
    }

    @Test
    public void testReshapeVjp() {
        // grad of reshape is reshape back
        Function<NDArray, NDArray> fn = x -> x.reshape(new Shape(2, 3));
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(3, 2));
        NDArray g = JAX.grad(fn).apply(x);
        // gradient of sum through reshape: all ones
        assertArrayEquals(new int[]{3, 2}, g.shape().dimensions());
    }

    // ------------------------------------------------------------------
    // transpose
    // ------------------------------------------------------------------

    @Test
    public void testTransposeForward() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        float[] data = {1, 2, 3, 4, 5, 6};
        NDArray x = new ConcreteNDArray(data, new Shape(2, 3));
        NDArray t = x.transpose();
        assertArrayEquals(new int[]{3, 2}, t.shape().dimensions());
        float[] out = t.toFloatArray();
        assertEquals(1f, out[0], 1e-6f);
        assertEquals(4f, out[1], 1e-6f);
        assertEquals(2f, out[2], 1e-6f);
        assertEquals(5f, out[3], 1e-6f);
        assertEquals(3f, out[4], 1e-6f);
        assertEquals(6f, out[5], 1e-6f);
    }

    @Test
    public void testTransposeVjp() {
        Function<NDArray, NDArray> fn = x -> x.transpose().reshape(new Shape(6));
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new int[]{2, 3}, g.shape().dimensions());
    }

    // ------------------------------------------------------------------
    // pad
    // ------------------------------------------------------------------

    @Test
    public void testPadForward() {
        float[] data = {1, 2, 3, 4};
        NDArray x = new ConcreteNDArray(data, new Shape(2, 2));
        // Pad 1 on each side of each axis -> (4, 4)
        int[][] padding = {{1, 1}, {1, 1}};
        NDArray padded = x.pad(padding);
        assertArrayEquals(new int[]{4, 4}, padded.shape().dimensions());
        float[] out = padded.toFloatArray();
        // Center 2x2 should match original
        assertEquals(0f, out[0], 1e-6f);   // top-left corner
        assertEquals(1f, out[5], 1e-6f);   // row 1, col 1
        assertEquals(2f, out[6], 1e-6f);   // row 1, col 2
        assertEquals(3f, out[9], 1e-6f);   // row 2, col 1
        assertEquals(4f, out[10], 1e-6f);  // row 2, col 2
    }

    @Test
    public void testPadVjp() {
        // padding {{0,1},{1,0}} on shape (2,2) -> (3,3) = 9 elements
        Function<NDArray, NDArray> fn = x -> x.pad(new int[][]{{0, 1}, {1, 0}}).reshape(new Shape(9));
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(2, 2));
        NDArray g = JAX.grad(fn).apply(x);
        assertArrayEquals(new int[]{2, 2}, g.shape().dimensions());
    }

    // ------------------------------------------------------------------
    // concatenate / stack
    // ------------------------------------------------------------------

    @Test
    public void testConcatenateAxis0() {
        NDArray a = new ConcreteNDArray(new float[]{1, 2}, new Shape(1, 2));
        NDArray b = new ConcreteNDArray(new float[]{3, 4}, new Shape(1, 2));
        NDArray c = Numpy.concatenate(List.of(a, b), 0);
        assertArrayEquals(new int[]{2, 2}, c.shape().dimensions());
        assertArrayEquals(new float[]{1, 2, 3, 4}, c.toFloatArray(), 1e-6f);
    }

    @Test
    public void testConcatenateAxis1() {
        NDArray a = new ConcreteNDArray(new float[]{1, 2}, new Shape(2, 1));
        NDArray b = new ConcreteNDArray(new float[]{3, 4}, new Shape(2, 1));
        NDArray c = Numpy.concatenate(List.of(a, b), 1);
        assertArrayEquals(new int[]{2, 2}, c.shape().dimensions());
        assertArrayEquals(new float[]{1, 3, 2, 4}, c.toFloatArray(), 1e-6f);
    }

    @Test
    public void testStack() {
        NDArray a = new ConcreteNDArray(new float[]{1, 2, 3}, new Shape(3));
        NDArray b = new ConcreteNDArray(new float[]{4, 5, 6}, new Shape(3));
        NDArray s = Numpy.stack(List.of(a, b), 0);
        assertArrayEquals(new int[]{2, 3}, s.shape().dimensions());
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, s.toFloatArray(), 1e-6f);
    }

    @Test
    public void testConcatVjpSplitsGradient() {
        // VJP of concat should split gradient back to each input
        NDArray a = new ConcreteNDArray(new float[]{1, 2, 3}, new Shape(3));
        NDArray b = new ConcreteNDArray(new float[]{4, 5, 6}, new Shape(3));
        // fn(a) = sum(concat([a, const_b], 0))
        Function<NDArray, NDArray> fn = x -> Numpy.concatenate(List.of(x, b), 0).reshape(new Shape(6));
        NDArray g = JAX.grad(fn).apply(a);
        // gradient w.r.t. a should be all-ones of shape (3,)
        assertArrayEquals(new int[]{3}, g.shape().dimensions());
        assertArrayEquals(new float[]{1, 1, 1}, g.toFloatArray(), 1e-6f);
    }

    // ------------------------------------------------------------------
    // scatter_add
    // ------------------------------------------------------------------

    @Test
    public void testScatterAddForward() {
        NDArray target = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(4));
        NDArray indices = new ConcreteNDArray(new int[]{0, 2, 2}, new Shape(3));
        NDArray updates = new ConcreteNDArray(new float[]{10, 20, 30}, new Shape(3));
        NDArray result = Lax.scatterAdd(target, indices, updates);
        assertArrayEquals(new int[]{4}, result.shape().dimensions());
        float[] out = result.toFloatArray();
        assertEquals(11f, out[0], 1e-6f);  // 1 + 10
        assertEquals(2f,  out[1], 1e-6f);  // unchanged
        assertEquals(53f, out[2], 1e-6f);  // 3 + 20 + 30
        assertEquals(4f,  out[3], 1e-6f);  // unchanged
    }

    @Test
    public void testScatterAddDtypeConserved() {
        NDArray target = new ConcreteNDArray(new float[]{0, 0, 0}, new Shape(3));
        NDArray indices = new ConcreteNDArray(new int[]{1}, new Shape(1));
        NDArray updates = new ConcreteNDArray(new float[]{5}, new Shape(1));
        NDArray result = Lax.scatterAdd(target, indices, updates);
        assertEquals(DType.FLOAT32, result.dtype());
    }
}
