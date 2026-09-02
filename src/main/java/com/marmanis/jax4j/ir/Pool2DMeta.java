package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#MAX_POOL_2D} and {@link Primitive#AVG_POOL_2D}
 * primitives: pool window, stride, padding, and which reduction (max vs mean)
 * the equation performs.
 *
 * <p>{@code poolSize} is {@code [pH, pW]}; {@code strides} is
 * {@code [strideH, strideW]}; {@code padding} is either {@code "valid"} or
 * {@code "same"}. {@code isMax} disambiguates the two pooling flavors so a
 * shared decoder can dispatch on it.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Pool2DMeta(int[] poolSize, int[] strides, String padding, boolean isMax) {}
