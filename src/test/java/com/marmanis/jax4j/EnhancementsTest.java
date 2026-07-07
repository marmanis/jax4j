package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for JDK-25-era enhancements: value_and_grad, checkpoint, toString, parallel ops.
 */
class EnhancementsTest {

    // ── value_and_grad ─────────────────────────────────────────────────────────

    @Test
    void valueAndGrad_returnsBothValueAndGrad() {
        // f(x) = x^2 + x => f(3) = 12, f'(3) = 2*3+1 = 7
        Function<NDArray, NDArray[]> vg = JAX.value_and_grad(x -> x.mul(x).add(x));
        NDArray x = new ConcreteNDArray(new float[]{3f}, new Shape(1));
        NDArray[] result = vg.apply(x);
        assertEquals(12f, result[0].toFloatArray()[0], 1e-5f, "value");
        assertEquals(7f,  result[1].toFloatArray()[0], 1e-5f, "grad");
    }

    @Test
    void valueAndGrad_matchesGradAndForward() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).mul(x); // f(x) = x^3
        Function<NDArray, NDArray[]> vg = JAX.value_and_grad(fn);
        NDArray x = new ConcreteNDArray(new float[]{2f}, new Shape(1));

        NDArray[] result = vg.apply(x);
        float value = result[0].toFloatArray()[0];
        float grad  = result[1].toFloatArray()[0];

        // f(2) = 8, f'(2) = 3*4 = 12
        assertEquals(8f,  value, 1e-5f, "value");
        assertEquals(12f, grad,  1e-5f, "grad");
    }

    // ── checkpoint (gradient rematerialization) ────────────────────────────────

    @Test
    void checkpoint_eagerModeIsTransparent() {
        // Outside grad, checkpoint is a no-op
        Function<NDArray, NDArray> fn = JAX.checkpoint(x -> x.mul(x));
        NDArray x = new ConcreteNDArray(new float[]{3f}, new Shape(1));
        assertEquals(9f, fn.apply(x).toFloatArray()[0], 1e-5f);
    }

    @Test
    void checkpoint_gradientsCorrect() {
        // grad(checkpoint(x -> x^2)) should equal grad(x -> x^2) = 2x
        Function<NDArray, NDArray> gradFn = JAX.grad(JAX.checkpoint(x -> x.mul(x)));
        NDArray x = new ConcreteNDArray(new float[]{4f}, new Shape(1));
        assertEquals(8f, gradFn.apply(x).toFloatArray()[0], 1e-5f);
    }

    @Test
    void checkpoint_composesWithOtherOps() {
        // f(x) = checkpoint(x -> x^2) + x; f'(x) = 2x + 1
        Function<NDArray, NDArray[]> vg = JAX.value_and_grad(
            x -> JAX.checkpoint((NDArray a) -> a.mul(a)).apply(x).add(x));
        NDArray x = new ConcreteNDArray(new float[]{3f}, new Shape(1));
        NDArray[] result = vg.apply(x);
        assertEquals(12f, result[0].toFloatArray()[0], 1e-5f, "value");
        assertEquals(7f,  result[1].toFloatArray()[0], 1e-5f, "grad");
    }

    // ── NDArray.toString() ─────────────────────────────────────────────────────

    @Test
    void toString_showsValues() {
        NDArray a = new ConcreteNDArray(new float[]{1f, 2f, 3f}, new Shape(3));
        String s = a.toString();
        assertTrue(s.startsWith("Array("), "starts with Array(");
        assertTrue(s.contains("shape="), "contains shape=");
        assertTrue(s.contains("dtype="), "contains dtype=");
        assertTrue(s.contains("1.0"), "contains data");
    }

    @Test
    void toString_float64() {
        NDArray a = new ConcreteNDArray(new double[]{1.5, 2.5}, new Shape(2));
        String s = a.toString();
        assertTrue(s.contains("float64"));
        assertTrue(s.contains("1.5"));
    }

    // ── broadcastIndex short-circuit ───────────────────────────────────────────

    @Test
    void broadcastIndex_sameSapeReturnsIdentity() {
        Shape s = new Shape(3, 4);
        // For same-shape, should return outFlatIndex unchanged
        for (int i = 0; i < 12; i++) {
            assertEquals(i, s.broadcastIndex(s, i));
        }
    }

    // ── parallel elementwise (functional correctness; timing is not asserted) ──

    @Test
    void parallelElementwise_largeArrayCorrect() {
        int n = 100_000;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2f; }
        NDArray x = new ConcreteNDArray(a, new Shape(n));
        NDArray y = new ConcreteNDArray(b, new Shape(n));
        float[] result = x.add(y).toFloatArray();
        assertEquals((n - 1) + 2f, result[n - 1], 1e-3f); // (n-1) + 2
    }
}
