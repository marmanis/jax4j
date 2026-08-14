package com.marmanis.jax4j.keras.metrics;

import com.marmanis.jax4j.core.NDArray;

/**
 * A stateful Keras-style metric that accumulates statistics across batches
 * and reports a running scalar via {@link #result()}. Reset with
 * {@link #reset()} at epoch boundaries.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Metric {
    String name();
    void reset();
    void update(NDArray yTrue, NDArray yPred);
    float result();
}
