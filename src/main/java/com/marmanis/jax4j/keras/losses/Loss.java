package com.marmanis.jax4j.keras.losses;

import com.marmanis.jax4j.core.NDArray;

/**
 * A Keras-style loss: takes ground-truth and predicted arrays, returns a
 * scalar {@link NDArray}. Composed from differentiable primitives so it works
 * transparently under {@code Grad.gradTree}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
@FunctionalInterface
public interface Loss {
    NDArray call(NDArray yTrue, NDArray yPred);
}
