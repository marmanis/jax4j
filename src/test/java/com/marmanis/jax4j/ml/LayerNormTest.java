package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.LayerNorm;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LayerNormTest {

    @Test
    public void normalizesLastAxisToZeroMeanUnitVariance() {
        LayerNorm ln = LayerNorm.init(4);
        NDArray x = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f, 10f, 12f, 14f, 16f}, new Shape(2, 4));
        NDArray y = ln.apply(x);
        float[] out = y.toFloatArray();
        for (int row = 0; row < 2; row++) {
            float mean = 0f;
            for (int j = 0; j < 4; j++) mean += out[row * 4 + j];
            mean /= 4f;
            float var = 0f;
            for (int j = 0; j < 4; j++) {
                float d = out[row * 4 + j] - mean;
                var += d * d;
            }
            var /= 4f;
            assertEquals(0f, mean, 1e-4);
            assertEquals(1f, var, 1e-3);
        }
    }
}
