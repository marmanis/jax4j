package com.marmanis.jax4j.ir;

/**
 * Metadata for a {@link Primitive#CHECKPOINT} equation.
 * Holds the sub-Jaxpr that will be re-executed (rematerialized) during the
 * backward pass instead of storing its intermediate activations.
 */
public record CheckpointMeta(Jaxpr subJaxpr) {}
