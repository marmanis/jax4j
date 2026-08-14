package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Dense;
import com.marmanis.jax4j.keras.losses.Losses;
import com.marmanis.jax4j.keras.optimizers.Adam;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class FitTest {

    @Test
    public void trivialRegressionLossDecreases() {
        int n = 64;
        float[] xd = new float[n];
        float[] yd = new float[n];
        for (int i = 0; i < n; i++) {
            xd[i] = i / 32f - 1f;
            yd[i] = 2f * xd[i];
        }
        NDArray x = new ConcreteNDArray(xd, new Shape(n, 1));
        NDArray y = new ConcreteNDArray(yd, new Shape(n, 1));

        Sequential model = Sequential.of(new Shape(1))
            .add(new Dense(8, "tanh"))
            .add(new Dense(1));
        model.compile(new Adam(5e-2f), Losses.mse());
        History h = model.fit(x, y, 30, 16, List.of());
        List<Float> losses = h.get("loss");
        assertTrue(losses.size() == 30, "expected 30 epochs, got " + losses.size());
        float first = losses.get(0);
        float last = losses.get(losses.size() - 1);
        assertTrue(last < first * 0.5f, "loss did not decrease enough: " + first + " -> " + last);
    }
}
