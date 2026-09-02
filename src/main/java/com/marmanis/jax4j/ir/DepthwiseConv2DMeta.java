package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#DEPTHWISE_CONV_2D} primitive: convolution
 * stride, padding mode, and channel depth multiplier. The depthwise kernel
 * shape (rank-4, {@code [kH, kW, inC, depthMultiplier]}) lives on the
 * equation's second input {@link Var}.
 *
 * <p>{@code strides} is {@code [strideH, strideW]}. {@code padding} is
 * either {@code "valid"} or {@code "same"} (Keras convention). Each input
 * channel is convolved with its own {@code depthMultiplier} filters; output
 * channels {@code = inC * depthMultiplier}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record DepthwiseConv2DMeta(int[] strides, String padding, int depthMultiplier) {}
