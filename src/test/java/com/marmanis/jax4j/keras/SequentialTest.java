package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Dense;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SequentialTest {

    @Test
    public void threeLayerMlpParamCountAndForward() {
        Sequential model = Sequential.of(new Shape(8))
            .add(new Dense(16, "relu"))
            .add(new Dense(8, "relu"))
            .add(new Dense(3));
        // 8*16 + 16 + 16*8 + 8 + 8*3 + 3 = 128 + 16 + 128 + 8 + 24 + 3 = 307
        assertEquals(307, model.countParams());

        NDArray x = Random.uniform(new PRNGKey(1L), new Shape(4, 8));
        NDArray y = model.call(x);
        assertEquals(new Shape(4, 3), y.shape());
    }

    @Test
    public void layerNamesAreSnakeCased() {
        Layer.resetNameCounters();
        Sequential model = Sequential.of(new Shape(4))
            .add(new Dense(4))
            .add(new Dense(4));
        assertEquals("dense", model.layers().get(0).getName());
        assertEquals("dense_1", model.layers().get(1).getName());
    }

    @Test
    public void removeLastShrinksTheStack() {
        Sequential model = Sequential.of(new Shape(4))
            .add(new Dense(8))
            .add(new Dense(3));
        assertEquals(2, model.layers().size());
        model.removeLast();
        assertEquals(1, model.layers().size());
        NDArray y = model.call(Random.uniform(new PRNGKey(0), new Shape(2, 4)));
        assertEquals(new Shape(2, 8), y.shape());
        assertTrue(model.countParams() > 0);
    }
}
