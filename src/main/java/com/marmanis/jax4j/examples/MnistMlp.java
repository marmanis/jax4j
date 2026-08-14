package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Modules;
import com.marmanis.jax4j.ml.loss.Losses;
import com.marmanis.jax4j.ml.nn.ActivationFn;
import com.marmanis.jax4j.ml.nn.MLP;
import com.marmanis.jax4j.ml.optim.Adam;
import com.marmanis.jax4j.ml.optim.Optimizer;
import com.marmanis.jax4j.ml.io.Serialization;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Minimal end-to-end demo of {@code com.marmanis.jax4j.ml}: build an MLP,
 * take a few Adam steps on synthetic data, save/load, and verify predictions
 * round-trip exactly.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class MnistMlp {

    public static void main(String[] args) throws Exception {
        ExampleBackend.selectFromArgs(args);

        PRNGKey key = PRNGKey.key(0);
        PRNGKey[] keys = Random.split(key, 3);

        MLP model = MLP.init(keys[0], 784, 10, 128, 3, ActivationFn.RELU);
        Adam optim = Adam.defaults(1e-3f);
        Adam.State optState = optim.init(model);

        int batch = 32;
        NDArray x = Random.normal(keys[1], new Shape(batch, 784));
        NDArray y = Random.uniform(keys[2], new Shape(batch, 10));

        for (int step = 0; step < 5; step++) {
            NDArray loss = Losses.mse(model.apply(x), y);
            MLP grads = Modules.filterGrad(m -> Losses.mse(m.apply(x), y), model);
            Optimizer.Update<MLP, Adam.State> upd = optim.update(model, grads, optState);
            model = upd.newModule();
            optState = upd.newState();
            System.out.printf("step %d loss=%.4f%n", step, loss.toFloatArray()[0]);
        }

        Path tmp = Files.createTempFile("mnist_mlp_", ".jx4j");
        Serialization.save(model, tmp);
        MLP loaded = Serialization.load(model, tmp);
        NDArray a = model.apply(x);
        NDArray b = loaded.apply(x);
        float maxDiff = 0f;
        float[] av = a.toFloatArray();
        float[] bv = b.toFloatArray();
        for (int i = 0; i < av.length; i++) maxDiff = Math.max(maxDiff, Math.abs(av[i] - bv[i]));
        System.out.printf("save/load round-trip max diff: %.2e%n", maxDiff);
        Files.deleteIfExists(tmp);
    }
}
