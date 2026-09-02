package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#GRID_SAMPLE_2D} primitive: how out-of-range
 * sample coordinates are handled.
 *
 * <p>{@code paddingMode} is either {@code "zeros"} (out-of-range corners
 * contribute 0) or {@code "border"} (sample coordinates are clamped to the
 * nearest valid pixel index).
 *
 * <p>The input image ({@code [B, H, W, C]}) and the sample grid
 * ({@code [B, outH, outW, 2]}, storing {@code (y, x)} in pixel coordinates)
 * live on the equation's two input {@link Var}s.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record GridSample2DMeta(String paddingMode) {}
