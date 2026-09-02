package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@code SLICE} primitive: per-axis {@code starts}, {@code stops}, and
 * {@code steps} arrays (all of length = input rank). {@code steps[i] > 0} (positive strides
 * only in v1). {@code stops[i] == -1} means "to end of axis" (sentinel decoded eagerly).
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record SliceMeta(int[] starts, int[] stops, int[] steps) {}
