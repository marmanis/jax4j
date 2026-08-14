package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Dropout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DropoutTest {

    @Test
    public void inferenceIsIdentity() {
        Dropout d = new Dropout(0.5f);
        d.build(new Shape(8), new PRNGKey(0));
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new Shape(8));
        NDArray y = d.call(x, null, false);
        assertArrayEquals(x.toFloatArray(), y.toFloatArray(), 1e-6f);
    }

    @Test
    public void trainingModeDropsSomeElements() {
        Dropout d = new Dropout(0.5f);
        d.build(new Shape(2048), new PRNGKey(0));
        float[] input = new float[2048];
        for (int i = 0; i < input.length; i++) input[i] = 1f;
        NDArray x = new ConcreteNDArray(input, new Shape(2048));
        NDArray y = d.call(x, null, true);
        int zeros = 0;
        for (float v : y.toFloatArray()) if (v == 0f) zeros++;
        // Roughly half should be zeroed; be generous with the bounds.
        assertTrue(zeros > 500 && zeros < 1500, "unexpected drop count: " + zeros);
    }
}
