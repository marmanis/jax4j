package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.Dropout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DropoutTest {

    @Test
    public void trainingFalseIsIdentity() {
        Dropout d = new Dropout(0.5f);
        float[] xData = {1f, 2f, 3f, 4f};
        NDArray x = new ConcreteNDArray(xData, new Shape(4));
        NDArray y = d.apply(x, PRNGKey.key(0), false);
        assertArrayEquals(xData, y.toFloatArray(), 1e-6f);
    }

    @Test
    public void trainingZeroesApproxRateFraction() {
        Dropout d = new Dropout(0.5f);
        int n = 10_000;
        float[] xData = new float[n];
        for (int i = 0; i < n; i++) xData[i] = 1f;
        NDArray x = new ConcreteNDArray(xData, new Shape(n));
        NDArray y = d.apply(x, PRNGKey.key(42), true);
        float[] out = y.toFloatArray();
        int zeros = 0;
        for (float v : out) if (v == 0f) zeros++;
        double frac = zeros / (double) n;
        assertTrue(frac > 0.45 && frac < 0.55,
            "expected ~0.5 zero fraction, got " + frac);
    }
}
