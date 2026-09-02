package com.marmanis.jax4j.ir;

/**
 * Metadata for the {@link Primitive#CONV3D} primitive: convolution stride and
 * padding mode across three spatial axes. The kernel shape (and therefore the
 * kernel dimensions) live on the equation's second input {@link Var}.
 *
 * <p>{@code strides} is {@code [strideD, strideH, strideW]}. {@code padding}
 * is either {@code "valid"} or {@code "same"} (Keras convention, per-axis
 * asymmetric split when the total is odd).
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Conv3DMeta(int[] strides, String padding) {}
