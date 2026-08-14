package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;

/**
 * Layer normalization over the last axis (v1). Standardizes each element's
 * feature vector to zero mean and unit variance, then applies a learned
 * per-feature affine {@code y = scale * xhat + shift}.
 *
 * <p>TODO: honor {@code normDim} for normalizing over more than one trailing
 * axis; currently only the last axis is supported.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record LayerNorm(NDArray scale, NDArray shift, @Static int normDim, @Static float eps) implements Module {

    public static LayerNorm init(int features) {
        return init(features, 1e-5f);
    }

    public static LayerNorm init(int features, float eps) {
        float[] onesData = new float[features];
        for (int i = 0; i < features; i++) onesData[i] = 1f;
        NDArray scale = new ConcreteNDArray(onesData, new Shape(features));
        NDArray shift = new ConcreteNDArray(new float[features], new Shape(features));
        return new LayerNorm(scale, shift, 1, eps);
    }

    public NDArray apply(NDArray x) {
        NDArray mean = x.mean(-1, true);
        NDArray centered = x.sub(mean);
        NDArray var = centered.mul(centered).mean(-1, true);
        NDArray epsArr = new ConcreteNDArray(new float[]{eps}, new Shape(1));
        NDArray halfNeg = new ConcreteNDArray(new float[]{-0.5f}, new Shape(1));
        // 1 / sqrt(var + eps) = exp(-0.5 * log(var + eps)) — differentiable through EXP/LOG/MUL/ADD.
        NDArray invStd = var.add(epsArr).log().mul(halfNeg).exp();
        return centered.mul(invStd).mul(scale).add(shift);
    }
}
