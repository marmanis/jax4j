package com.marmanis.jax4j.keras.metrics;

import com.marmanis.jax4j.core.NDArray;

/**
 * Alias for {@link Accuracy} that always interprets {@code yTrue} as class
 * indices (mirroring {@code keras.metrics.SparseCategoricalAccuracy}).
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class SparseCategoricalAccuracy implements Metric {
    private final Accuracy inner = new Accuracy();

    @Override public String name() { return "sparse_categorical_accuracy"; }
    @Override public void reset() { inner.reset(); }
    @Override public void update(NDArray yTrue, NDArray yPred) { inner.update(yTrue, yPred); }
    @Override public float result() { return inner.result(); }
}
