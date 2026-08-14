package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-layer perceptron: {@code depth} {@link Linear} layers separated by
 * {@code activation}. Activation is applied between hidden layers only, not
 * after the final projection.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record MLP(List<Linear> layers, @Static ActivationFn activation) implements Module {

    public static MLP init(PRNGKey key, int inDim, int outDim, int hiddenDim, int depth, ActivationFn activation) {
        if (depth < 1) throw new IllegalArgumentException("depth must be >= 1, got " + depth);
        PRNGKey[] keys = Random.split(key, depth);
        List<Linear> layers = new ArrayList<>(depth);
        for (int i = 0; i < depth; i++) {
            int in = (i == 0) ? inDim : hiddenDim;
            int out = (i == depth - 1) ? outDim : hiddenDim;
            layers.add(Linear.init(keys[i], in, out));
        }
        return new MLP(layers, activation);
    }

    public NDArray apply(NDArray x) {
        NDArray h = x;
        int last = layers.size() - 1;
        for (int i = 0; i < layers.size(); i++) {
            h = layers.get(i).apply(h);
            if (i != last) h = activation.apply(h);
        }
        return h;
    }
}
