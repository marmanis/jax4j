package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Input;
import com.marmanis.jax4j.keras.KerasTensor;
import com.marmanis.jax4j.keras.Model;
import com.marmanis.jax4j.keras.layers.Activation;
import com.marmanis.jax4j.keras.layers.Dense;
import com.marmanis.jax4j.keras.losses.Losses;
import com.marmanis.jax4j.keras.optimizers.Adam;

import java.util.List;

/**
 * Functional-API example: an MLP with a residual connection built by calling
 * layers on symbolic {@link KerasTensor}s and adding two branches together.
 * Uses synthetic random data.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class KerasFunctionalDemo {
    private KerasFunctionalDemo() {}

    public static void main(String[] args) {
        PRNGKey key = new PRNGKey(0L);
        PRNGKey[] sub = Random.split(key, 3);
        int nTrain = 128;
        NDArray xTrain = Random.uniform(sub[0], new Shape(nTrain, 64));
        NDArray yTrain = Random.uniform(sub[1], new Shape(nTrain, 10));

        KerasTensor input = Input.input(new Shape(64));
        KerasTensor h = new Dense(64, "relu").apply(input);
        KerasTensor skip = new Dense(64).apply(h);
        KerasTensor merged = new Activation("relu").apply(input.add(skip));
        KerasTensor output = new Dense(10, "softmax").apply(merged);

        Model model = new Model(input, output);
        model.compile(new Adam(1e-3f), Losses.mse());
        model.summary();
        model.fit(xTrain, yTrain, 1, 32);
        System.out.println("prediction[0][:5]:");
        float[] out = model.predict(xTrain, 32).toFloatArray();
        for (int i = 0; i < 5; i++) System.out.printf("  %.4f%n", out[i]);
    }
}
