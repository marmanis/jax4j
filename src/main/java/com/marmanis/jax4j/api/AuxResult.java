package com.marmanis.jax4j.api;

/**
 * Container holding a primary value (such as a scalar loss or gradient)
 * alongside auxiliary data (such as intermediate stats, metrics, or activations).
 *
 * <p>Mirrors JAX's {@code has_aux=True} pattern where transformations differentiate
 * strictly with respect to the primary output while passing the auxiliary output through.
 *
 * @param <V> Type of primary value (e.g. {@code NDArray}, {@code PyTree}, or arrays thereof)
 * @param <A> Type of auxiliary data
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record AuxResult<V, A>(V value, A aux) {

    public static <V, A> AuxResult<V, A> of(V value, A aux) {
        return new AuxResult<>(value, aux);
    }
}
