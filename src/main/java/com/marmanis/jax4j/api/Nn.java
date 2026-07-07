package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;

/**
 * Activation functions mirroring {@code jax.nn}. {@code tanh}/{@code relu}/
 * {@code sigmoid} are core, differentiable {@link NDArray} primitives (called
 * directly, e.g. {@code x.tanh()}); {@link #softmax} is a composite built
 * from {@code exp}/{@code sum(axis, keepDims)}/{@code div} — the axis-aware
 * {@code sum} keeps the row-sum differentiable via {@code SUM_AXIS}'s VJP
 * rule without needing a dedicated softmax primitive.
 */
public final class Nn {
    private Nn() {}

    /** Softmax over the last axis, mirroring {@code jax.nn.softmax(x, axis=-1)}. */
    public static NDArray softmax(NDArray x) {
        NDArray expX = x.exp();
        NDArray rowSums = expX.sum(-1, true);
        return expX.div(rowSums);
    }

    /**
     * Inverted dropout, mirroring common Flax/PyTorch {@code dropout} semantics:
     * at train time, zeroes each element independently with probability {@code
     * rate} and rescales survivors by {@code 1 / (1 - rate)} so the expected
     * activation magnitude is unchanged (no rescaling needed at eval time).
     * {@code training=false} is a no-op pass-through.
     *
     * <p>Composed entirely from {@link Random#bernoulli} (a constant mask,
     * explicitly cast from its real {@code DType.BOOL} to FLOAT32 — jax4j's
     * arithmetic ops require matching FLOAT32 operands, so this cast is the
     * principled stand-in for what used to be an implicit float-valued mask)
     * and the existing differentiable {@code mul}/{@code div} primitives — no
     * new {@code Primitive} or VJP rule needed; the mask just rides along as a
     * captured constant when this runs inside a traced function.
     */
    public static NDArray dropout(NDArray x, PRNGKey key, float rate, boolean training) {
        if (!training) return x;
        NDArray keepMask = Random.bernoulli(key, 1f - rate, x.shape()).astype(DType.FLOAT32);
        NDArray keepProb = new ConcreteNDArray(new float[]{1f - rate}, new Shape(1));
        return x.mul(keepMask).div(keepProb);
    }
}
