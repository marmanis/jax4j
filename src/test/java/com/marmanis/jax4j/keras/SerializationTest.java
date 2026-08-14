package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.keras.layers.Dense;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class SerializationTest {

    @Test
    public void saveAndLoadRoundTripPreservesPredictions(@TempDir Path tmp) throws Exception {
        Sequential a = Sequential.of(new Shape(4))
            .add(new Dense(8, "relu"))
            .add(new Dense(3));
        NDArray x = Random.uniform(new PRNGKey(0), new Shape(2, 4));
        float[] predA = a.call(x).toFloatArray();

        Path path = tmp.resolve("weights.krs");
        a.saveWeights(path);

        Sequential b = Sequential.of(new Shape(4))
            .add(new Dense(8, "relu"))
            .add(new Dense(3));
        b.loadWeights(path);
        float[] predB = b.call(x).toFloatArray();

        assertArrayEquals(predA, predB, 1e-6f);
    }
}
