package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.api.Vmap;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for comparisons, max/min, clip, where, and argmax/argmin: selection
 * and branching primitives that must stay traceable/differentiable rather
 * than dropping out of the API (comparisons/argmax give zero gradient by
 * convention; max/min route gradient to the winning operand; where is
 * composed from existing differentiable ops).
 */
public class SelectionOpsTest {

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }

    private static NDArray vec(float... vs) {
        return new ConcreteNDArray(vs, new Shape(vs.length));
    }

    @Test
    public void comparisonsProduceOneZeroMasks() {
        NDArray a = vec(1f, 2f, 3f);
        NDArray b = vec(3f, 2f, 1f);
        assertArrayEquals(new boolean[]{false, false, true}, a.gt(b).toBoolArray());
        assertArrayEquals(new boolean[]{false, true, true}, a.ge(b).toBoolArray());
        assertArrayEquals(new boolean[]{true, false, false}, a.lt(b).toBoolArray());
        assertArrayEquals(new boolean[]{true, true, false}, a.le(b).toBoolArray());
        assertArrayEquals(new boolean[]{false, true, false}, a.eq(b).toBoolArray());
        assertArrayEquals(new boolean[]{true, false, true}, a.ne(b).toBoolArray());
        assertEquals(com.marmanis.jax4j.core.DType.BOOL, a.gt(b).dtype());
    }

    @Test
    public void maxMinForwardValues() {
        NDArray a = vec(1f, 5f, 3f);
        NDArray b = vec(4f, 2f, 3f);
        assertArrayEquals(new float[]{4, 5, 3}, a.max(b).toFloatArray(), 1e-6f);
        assertArrayEquals(new float[]{1, 2, 3}, a.min(b).toFloatArray(), 1e-6f);
    }

    @Test
    public void clipIsMaxThenMin() {
        NDArray x = vec(-2f, 0.5f, 5f);
        NDArray clipped = x.clip(scalar(0f), scalar(1f));
        assertArrayEquals(new float[]{0, 0.5f, 1}, clipped.toFloatArray(), 1e-6f);
    }

    @Test
    public void argmaxArgminAlongAxis() {
        // 2x3: [[1,5,3],[9,2,4]]
        NDArray x = new ConcreteNDArray(new float[]{1, 5, 3, 9, 2, 4}, new Shape(2, 3));
        assertArrayEquals(new int[]{1, 0}, x.argmax(1).toIntArray());
        assertArrayEquals(new int[]{0, 1}, x.argmin(1).toIntArray());
        assertEquals(com.marmanis.jax4j.core.DType.INT32, x.argmax(1).dtype());
    }

    @Test
    public void comparisonsHaveZeroGradientNotException() {
        Function<NDArray, NDArray> fn = x -> x.gt(scalar(0f));
        NDArray grad = JAX.grad(fn).apply(scalar(3f));
        assertEquals(0f, grad.toFloatArray()[0], 1e-6f);
    }

    @Test
    public void argmaxHasZeroGradient() {
        Function<NDArray, NDArray> fn = x -> x.argmax(0);
        NDArray grad = JAX.grad(fn).apply(vec(1f, 2f, 3f));
        assertArrayEquals(new float[]{0, 0, 0}, grad.toFloatArray(), 1e-6f);
    }

    @Test
    public void maxGradFlowsToWinner() {
        // f(a, b) = max(a, b), d/da = 1 if a wins else 0; tie goes to a.
        Function<NDArray, NDArray> fnA = a -> a.max(scalar(2f)).sum();
        assertEquals(1f, JAX.grad(fnA).apply(scalar(5f)).toFloatArray()[0], 1e-6f); // a=5 wins
        assertEquals(0f, JAX.grad(fnA).apply(scalar(1f)).toFloatArray()[0], 1e-6f); // a=1 loses
        assertEquals(1f, JAX.grad(fnA).apply(scalar(2f)).toFloatArray()[0], 1e-6f); // tie -> a wins
    }

    @Test
    public void whereSelectsAndIsDifferentiable() {
        // f(x) = where(x > 0, x^2, -x) ; d/dx for x>0 is 2x, for x<=0 is -1.
        Function<NDArray, NDArray> fn = x -> Numpy.where(x.gt(scalar(0f)), x.mul(x), x.mul(scalar(-1f)));

        assertEquals(9f, fn.apply(scalar(3f)).toFloatArray()[0], 1e-5f);
        assertEquals(4f, fn.apply(scalar(-4f)).toFloatArray()[0], 1e-5f);

        assertEquals(6f, JAX.grad(fn).apply(scalar(3f)).toFloatArray()[0], 1e-4f);
        assertEquals(-1f, JAX.grad(fn).apply(scalar(-4f)).toFloatArray()[0], 1e-4f);
    }

    @Test
    public void vmapComposesWithArgmax() {
        // Per-row argmax over a batch of rows.
        Function<NDArray, NDArray> rowArgmax = row -> row.argmax(0);
        NDArray batch = new ConcreteNDArray(new float[]{1, 5, 3, 9, 2, 4}, new Shape(2, 3));
        NDArray result = Vmap.vmap(rowArgmax).apply(batch);
        assertArrayEquals(new int[]{1, 0}, result.toIntArray());
    }
}
