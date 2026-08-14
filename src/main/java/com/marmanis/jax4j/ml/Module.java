package com.marmanis.jax4j.ml;

/**
 * Marker interface for immutable, record-based neural-network components.
 *
 * <p>Every {@code Module} is a Java {@code record} whose components are one of:
 * an {@link com.marmanis.jax4j.core.NDArray} parameter leaf, another
 * {@code Module} (nested), a {@code List<? extends Module>} or {@code Module[]}
 * (ordered composite), or a hyperparameter field marked with {@link Static}
 * (or otherwise non-parameter — primitives, boxed numbers, strings, enums).
 * {@link Modules} reflects on record components to expose the parameter set
 * as a {@link com.marmanis.jax4j.pytree.PyTree}.
 *
 * <p>This mirrors the design of Equinox in Python/JAX: models are just
 * immutable tree-structured values; every functional update produces a new
 * value with the same static fields and swapped-in parameter leaves.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Module {
}
