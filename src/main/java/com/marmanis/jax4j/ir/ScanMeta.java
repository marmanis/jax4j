package com.marmanis.jax4j.ir;

/**
 * Metadata for a {@code SCAN} equation: {@code stepFn} is a two-input
 * {@code (carry, x)} / two-output {@code (newCarry, y)} {@link Jaxpr},
 * applied once per leading-axis slice of the equation's {@code xs} input.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record ScanMeta(Jaxpr stepFn) {
}
