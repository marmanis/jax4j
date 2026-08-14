package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;

/**
 * Affine transform: {@code y = x . weight + bias}. Weight has shape
 * {@code [inFeatures, outFeatures]}; bias has shape {@code [outFeatures]}
 * and is broadcast across the leading batch dimensions.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Linear(NDArray weight, NDArray bias) implements Module {

    /**
     * Glorot/Xavier-uniform initialization; bias starts at zero.
     */
    public static Linear init(PRNGKey key, int inFeatures, int outFeatures) {
        NDArray w = Random.glorotUniform(key, new Shape(inFeatures, outFeatures), inFeatures, outFeatures);
        NDArray b = new ConcreteNDArray(new float[outFeatures], new Shape(outFeatures));
        return new Linear(w, b);
    }

    public NDArray apply(NDArray x) {
        return x.dot(weight).add(bias);
    }
}
