package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#CONV_2D_TRANSPOSE} primitive: transposed
 * convolution stride and padding mode. The kernel shape (rank-4,
 * {@code [kH, kW, filters, inC]} — note the swapped channel axes vs
 * {@link Primitive#CONV2D}, matching Keras convention) lives on the
 * equation's second input {@link Var}.
 *
 * <p>{@code strides} is {@code [strideH, strideW]}. {@code padding} is
 * either {@code "valid"} or {@code "same"} (Keras convention). Output
 * spatial size for {@code "valid"} is {@code (inH - 1) * strideH + kH};
 * for {@code "same"} it is {@code inH * strideH}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Conv2DTransposeMeta(int[] strides, String padding) {}
