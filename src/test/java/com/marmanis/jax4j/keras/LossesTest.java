package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.losses.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LossesTest {

    @Test
    public void mseOfEqualIsZero() {
        NDArray a = new ConcreteNDArray(new float[]{1f, 2f, 3f}, new Shape(3));
        float v = Losses.mse().call(a, a).toFloatArray()[0];
        assertEquals(0f, v, 1e-6f);
    }

    @Test
    public void categoricalCrossentropyOfPerfectIsSmall() {
        NDArray yTrue = new ConcreteNDArray(new float[]{1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
        NDArray yPred = new ConcreteNDArray(new float[]{0.999f, 5e-4f, 5e-4f, 5e-4f, 0.999f, 5e-4f}, new Shape(2, 3));
        float v = Losses.categoricalCrossentropy().call(yTrue, yPred).toFloatArray()[0];
        assertTrue(v < 0.01f, "expected near-zero loss, got " + v);
    }

    @Test
    public void sparseCategoricalCrossentropyRuns() {
        NDArray yPred = new ConcreteNDArray(new float[]{0.7f, 0.2f, 0.1f, 0.1f, 0.1f, 0.8f}, new Shape(2, 3));
        NDArray yTrue = new ConcreteNDArray(new int[]{0, 2}, new Shape(2));
        float v = Losses.sparseCategoricalCrossentropy().call(yTrue, yPred).toFloatArray()[0];
        assertTrue(v > 0f && v < 1f, "unexpected loss: " + v);
    }
}
