package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.loss.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LossesTest {

    @Test
    public void mseOfEqualArraysIsZero() {
        float[] data = {1f, 2f, 3f, 4f};
        NDArray a = new ConcreteNDArray(data, new Shape(4));
        NDArray b = new ConcreteNDArray(data.clone(), new Shape(4));
        assertEquals(0f, Losses.mse(a, b).toFloatArray()[0], 1e-6);
    }

    @Test
    public void softmaxCrossEntropyApproachesZeroForConfidentCorrect() {
        // logits strongly favor class 1; target one-hot = [0, 1]
        NDArray logits = new ConcreteNDArray(new float[]{-10f, 10f}, new Shape(1, 2));
        NDArray target = new ConcreteNDArray(new float[]{0f, 1f}, new Shape(1, 2));
        float loss = Losses.softmaxCrossEntropy(logits, target).toFloatArray()[0];
        assertTrue(loss < 1e-3, "expected near-zero loss for confident correct prediction, got " + loss);
    }
}
