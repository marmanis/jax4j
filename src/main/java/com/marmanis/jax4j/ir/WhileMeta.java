package com.marmanis.jax4j.ir;

/**
 * Metadata for a {@code WHILE} equation: {@code condFn} (state -> truthy
 * scalar) and {@code bodyFn} (state -> new state), each a single-input/
 * single-output {@link Jaxpr}.
 */
public record WhileMeta(Jaxpr condFn, Jaxpr bodyFn) {
}
