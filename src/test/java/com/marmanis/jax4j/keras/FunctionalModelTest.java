package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Activation;
import com.marmanis.jax4j.keras.layers.Dense;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FunctionalModelTest {

    @Test
    public void skipConnectionShapeAndForward() {
        KerasTensor input = Input.input(new Shape(16));
        KerasTensor h = new Dense(16, "relu").apply(input);
        KerasTensor skip = new Dense(16).apply(h);
        KerasTensor merged = new Activation("relu").apply(input.add(skip));
        KerasTensor out = new Dense(4).apply(merged);
        Model model = new Model(input, out);
        NDArray x = Random.uniform(new PRNGKey(2L), new Shape(3, 16));
        NDArray y = model.call(x);
        assertEquals(new Shape(3, 4), y.shape());
    }

    @Test
    public void layersBuiltInTopoOrder() {
        KerasTensor input = Input.input(new Shape(4));
        KerasTensor a = new Dense(4).apply(input);
        KerasTensor b = new Dense(4).apply(a);
        Model model = new Model(input, b);
        assertEquals(2, model.layers().size());
        for (Layer l : model.layers()) {
            org.junit.jupiter.api.Assertions.assertTrue(l.isBuilt());
        }
    }
}
