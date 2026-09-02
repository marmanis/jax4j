package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#CONV2D} primitive: convolution stride and
 * padding mode. The kernel shape (and therefore the kernel dimensions) live on
 * the equation's second input {@link Var}.
 *
 * <p>{@code strides} is {@code [strideH, strideW]}. {@code padding} is
 * either {@code "valid"} or {@code "same"} (Keras convention).
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Conv2DMeta(int[] strides, String padding) {}
