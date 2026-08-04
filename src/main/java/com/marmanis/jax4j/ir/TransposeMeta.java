package com.marmanis.jax4j.ir;

/** Metadata for the TRANSPOSE primitive: the axis permutation. */
/**
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */

public record TransposeMeta(int[] axes) {}
