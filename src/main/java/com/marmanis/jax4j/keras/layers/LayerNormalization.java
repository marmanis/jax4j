package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Layer normalization over the last axis with learned per-feature scale ({@code gamma})
 * and shift ({@code beta}). {@code invStd} is computed as {@code exp(-0.5*log(var+eps))}
 * to stay entirely inside jax4j's differentiable primitive set.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class LayerNormalization extends Layer {

    private final float epsilon;

    public LayerNormalization() { this(1e-5f, null); }
    public LayerNormalization(float epsilon) { this(epsilon, null); }
    public LayerNormalization(float epsilon, String name) { super(name); this.epsilon = epsilon; }

    @Override protected String defaultNamePrefix() { return "layer_normalization"; }

    @Override public Shape computeOutputShape(Shape inputShape) { return inputShape; }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {
        int[] d = inputShape.dimensions();
        int features = d[d.length - 1];
        float[] onesData = new float[features];
        java.util.Arrays.fill(onesData, 1f);
        params.put("gamma", new ConcreteNDArray(onesData, new Shape(features)));
        params.put("beta", new ConcreteNDArray(new float[features], new Shape(features)));
        paramNames.add("gamma");
        paramNames.add("beta");
    }

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        NDArray x = inputs.get(0);
        NDArray mean = x.mean(-1, true);
        NDArray centered = x.sub(mean);
        NDArray var = centered.mul(centered).mean(-1, true);
        NDArray epsArr = new ConcreteNDArray(new float[]{epsilon}, new Shape(1));
        NDArray halfNeg = new ConcreteNDArray(new float[]{-0.5f}, new Shape(1));
        NDArray invStd = var.add(epsArr).log().mul(halfNeg).exp();
        return centered.mul(invStd).mul(effectiveParams.get("gamma")).add(effectiveParams.get("beta"));
    }
}
