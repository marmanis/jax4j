package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.ml.Static;
import com.marmanis.jax4j.ml.Stochastic;

/**
 * Inverted dropout with drop probability {@link #rate}. Delegates to
 * {@link Nn#dropout}. When called via a Sequential without a key or with
 * training disabled, this is the identity.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Dropout(@Static float rate) implements Stochastic {

    @Override
    public NDArray apply(NDArray x, PRNGKey key, boolean training) {
        return Nn.dropout(x, key, rate, training);
    }

    /** No-op pass-through for inference-time use inside a deterministic Sequential. */
    public NDArray applyEval(NDArray x) {
        return x;
    }
}
