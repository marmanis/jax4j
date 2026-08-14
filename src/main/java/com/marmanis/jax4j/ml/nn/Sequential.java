package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Stochastic;

import java.util.List;

/**
 * Layer stack whose forward pass folds the input through {@link #layers} in
 * order. If any layer is {@link Stochastic}, callers must use
 * {@link #apply(NDArray, PRNGKey, boolean)} — the key is split so each
 * stochastic layer draws its own independent randomness.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Sequential(List<Module> layers) implements Module {

    public NDArray apply(NDArray x) {
        NDArray h = x;
        for (Module layer : layers) h = applyLayer(layer, h);
        return h;
    }

    /**
     * Threads a key through any {@link Stochastic} layers; deterministic
     * layers ignore the key.
     */
    public NDArray apply(NDArray x, PRNGKey key, boolean training) {
        int stochasticCount = 0;
        for (Module layer : layers) if (layer instanceof Stochastic) stochasticCount++;
        PRNGKey[] subkeys = stochasticCount == 0 ? new PRNGKey[0] : Random.split(key, stochasticCount);
        int idx = 0;
        NDArray h = x;
        for (Module layer : layers) {
            if (layer instanceof Stochastic s) {
                h = s.apply(h, subkeys[idx++], training);
            } else {
                h = applyLayer(layer, h);
            }
        }
        return h;
    }

    private static NDArray applyLayer(Module layer, NDArray x) {
        return switch (layer) {
            case Linear l -> l.apply(x);
            case MLP m -> m.apply(x);
            case LayerNorm ln -> ln.apply(x);
            case Sequential s -> s.apply(x);
            case Dropout d -> d.applyEval(x);
            default -> throw new IllegalStateException(
                "Sequential: layer " + layer.getClass().getSimpleName()
                + " has no deterministic apply; either use Sequential.apply(x, key, training) "
                + "or extend the dispatch here.");
        };
    }
}
