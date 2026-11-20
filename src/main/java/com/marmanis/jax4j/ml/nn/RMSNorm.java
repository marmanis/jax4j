package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;

/**
 * Root Mean Square Normalization (RMSNorm) over the last axis.
 *
 * <p>Unlike standard {@link LayerNorm}, RMSNorm scales the input by its root-mean-square
 * value without centering around the mean, significantly reducing computation while maintaining
 * equivalent model convergence and stability. It is the default normalization method across
 * state-of-the-art transformer architectures (such as LLaMA, Mistral, and Gemma).
 *
 * <p>Formula:
 * <pre>
 *   rms = sqrt(mean(x^2, axis=-1, keepdims=True) + eps)
 *   y = (x / rms) * weight
 * </pre>
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record RMSNorm(NDArray weight, @Static float eps) implements Module {

    public static RMSNorm init(int dim) {
        return init(dim, 1e-6f);
    }

    public static RMSNorm init(int dim, float eps) {
        float[] ones = new float[dim];
        for (int i = 0; i < dim; i++) {
            ones[i] = 1.0f;
        }
        NDArray weight = new ConcreteNDArray(ones, new Shape(dim));
        return new RMSNorm(weight, eps);
    }

    /**
     * Applies RMSNorm along the trailing dimension of {@code x}.
     */
    public NDArray apply(NDArray x) {
        NDArray sq = x.mul(x);
        NDArray meanSq = sq.mean(-1, true);
        NDArray epsArr = new ConcreteNDArray(new float[]{eps}, new Shape(1));
        NDArray halfNeg = new ConcreteNDArray(new float[]{-0.5f}, new Shape(1));

        // 1 / sqrt(meanSq + eps) = exp(-0.5 * log(meanSq + eps)) — fully differentiable across all backends
        NDArray rsqrt = meanSq.add(epsArr).log().mul(halfNeg).exp();
        return x.mul(rsqrt).mul(weight);
    }
}
