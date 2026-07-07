package com.marmanis.jax4j.core;

/**
 * An explicit, immutable pseudo-random key, mirroring {@code jax.random.PRNGKey}.
 * jax4j threads randomness explicitly through {@link com.marmanis.jax4j.api.Random}
 * rather than relying on hidden global RNG state: every sampling call takes a key
 * and is a pure function of it, and {@link com.marmanis.jax4j.api.Random#split}
 * derives independent child keys for use in different parts of a computation
 * (e.g. one subkey per layer's weight init, another for a dropout mask).
 *
 * <p>Deliberately a plain Java value, not an {@link NDArray} — it is never traced
 * and never appears inside a {@code Jaxpr}; sampling always executes eagerly. See
 * {@link com.marmanis.jax4j.api.Random} for why this keeps autodiff/vmap/pmap
 * integration free (the resulting array is captured as an ordinary constant).
 */
public record PRNGKey(long state) {

    /** Mirrors {@code jax.random.PRNGKey(seed)} / {@code jax.random.key(seed)}. */
    public static PRNGKey key(long seed) {
        return new PRNGKey(seed);
    }
}
