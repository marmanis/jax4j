package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;

/**
 * Sub-interface for {@link Module}s whose forward pass consumes an explicit
 * random key (e.g. {@link com.marmanis.jax4j.ml.nn.Dropout}). Deterministic
 * layers implement {@code NDArray apply(NDArray x)} directly.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Stochastic extends Module {
    NDArray apply(NDArray x, PRNGKey key, boolean training);
}
