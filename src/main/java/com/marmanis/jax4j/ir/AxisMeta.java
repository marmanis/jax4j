package com.marmanis.jax4j.ir;

/**
 * Metadata for {@code SUM_AXIS}/{@code MEAN_AXIS} equations: which axis was
 * reduced (always normalized to {@code [0, rank())}, never negative) and
 * whether the reduced dimension was kept as size 1 or dropped.
 */
public record AxisMeta(int axis, boolean keepDims) {
}
