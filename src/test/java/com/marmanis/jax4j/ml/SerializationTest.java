package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.io.Serialization;
import com.marmanis.jax4j.ml.nn.ActivationFn;
import com.marmanis.jax4j.ml.nn.MLP;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class SerializationTest {

    @Test
    public void saveLoadRoundTripPreservesPredictions() throws Exception {
        MLP mlp = MLP.init(PRNGKey.key(11), 4, 3, 5, 2, ActivationFn.RELU);
        NDArray x = new ConcreteNDArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new Shape(1, 4));
        float[] yOrig = mlp.apply(x).toFloatArray();

        Path tmp = Files.createTempFile("jx4j_ser_", ".bin");
        try {
            Serialization.save(mlp, tmp);
            // Load into a fresh template with the same architecture but different weights.
            MLP template = MLP.init(PRNGKey.key(99), 4, 3, 5, 2, ActivationFn.RELU);
            MLP loaded = Serialization.load(template, tmp);
            float[] yLoaded = loaded.apply(x).toFloatArray();
            assertArrayEquals(yOrig, yLoaded, 1e-6f);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
