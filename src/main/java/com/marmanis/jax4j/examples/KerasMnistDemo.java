package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.History;
import com.marmanis.jax4j.keras.Sequential;
import com.marmanis.jax4j.keras.callbacks.ProgbarLogger;
import com.marmanis.jax4j.keras.layers.Dense;
import com.marmanis.jax4j.keras.layers.Dropout;
import com.marmanis.jax4j.keras.losses.Losses;
import com.marmanis.jax4j.keras.metrics.Accuracy;
import com.marmanis.jax4j.keras.optimizers.Adam;

import java.util.List;
import java.util.Map;

/**
 * The classic Keras MLP: {@code Sequential.of(...).add(...).add(...)} plus
 * {@code compile} / {@code fit} / {@code evaluate}. Uses synthetic random
 * data (no MNIST download required) so the point of the demo is the API,
 * not the training result.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class KerasMnistDemo {
    private KerasMnistDemo() {}

    public static void main(String[] args) {
        PRNGKey key = new PRNGKey(0L);
        PRNGKey[] sub = Random.split(key, 4);
        int nTrain = 256, nTest = 64;

        NDArray xTrain = Random.uniform(sub[0], new Shape(nTrain, 784), 0f, 1f);
        NDArray yTrain = oneHot(Random.permutation(sub[1], nTrain).toIntArray(), 10, nTrain);
        NDArray xTest = Random.uniform(sub[2], new Shape(nTest, 784), 0f, 1f);
        NDArray yTest = oneHot(Random.permutation(sub[3], nTest).toIntArray(), 10, nTest);

        Sequential model = Sequential.of(new Shape(784))
            .add(new Dense(128, "relu"))
            .add(new Dropout(0.2f))
            .add(new Dense(64, "relu"))
            .add(new Dense(10, "softmax"));

        model.compile(new Adam(1e-3f), Losses.categoricalCrossentropy(), List.of(new Accuracy()));
        model.summary();

        History h = model.fit(xTrain, yTrain, 2, 32, List.of(new ProgbarLogger()));
        Map<String, Float> results = model.evaluate(xTest, yTest, 32);
        System.out.println("test: " + results + " (trained " + h.get("loss").size() + " epochs)");
    }

    private static NDArray oneHot(int[] labels, int classes, int batch) {
        float[] out = new float[batch * classes];
        for (int i = 0; i < batch; i++) out[i * classes + (labels[i] % classes)] = 1f;
        return new com.marmanis.jax4j.core.ConcreteNDArray(out, new Shape(batch, classes));
    }
}
