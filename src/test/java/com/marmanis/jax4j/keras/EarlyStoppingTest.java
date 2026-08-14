package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.callbacks.EarlyStopping;
import com.marmanis.jax4j.keras.layers.Dense;
import com.marmanis.jax4j.keras.losses.Losses;
import com.marmanis.jax4j.keras.optimizers.SGD;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class EarlyStoppingTest {

    @Test
    public void stopsBeforeAllEpochsWhenLossPlateaus() {
        int n = 32;
        NDArray x = new ConcreteNDArray(new float[n], new Shape(n, 1));
        NDArray y = new ConcreteNDArray(new float[n], new Shape(n, 1));
        Sequential model = Sequential.of(new Shape(1))
            .add(new Dense(4))
            .add(new Dense(1));
        model.compile(new SGD(1e-4f), Losses.mse());
        History h = model.fit(x, y, 20, 8, List.of(new EarlyStopping("loss", 2, 1e-6f)));
        assertTrue(h.get("loss").size() < 20, "expected early stopping to trigger; got " + h.get("loss").size() + " epochs");
    }
}
