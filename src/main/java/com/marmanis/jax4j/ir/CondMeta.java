package com.marmanis.jax4j.ir;

/** Metadata for a {@code COND} equation: the two branches, each its own single-input/single-output {@link Jaxpr}. */
public record CondMeta(Jaxpr trueBranch, Jaxpr falseBranch) {
}
