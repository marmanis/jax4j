package com.marmanis.jax4j.ir;

/**
 * Metadata for the PAD primitive: per-dimension padding amounts.
 * {@code padding[i] = {before_i, after_i}} — number of zero-elements
 * to prepend and append to dimension {@code i}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record PadMeta(int[][] padding) {}
