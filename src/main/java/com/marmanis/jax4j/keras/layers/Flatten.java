package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Flattens a rank-{@code n} tensor to rank 2, preserving the leading batch
 * dimension: {@code [batch, d1, d2, ...] -> [batch, d1*d2*...]}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Flatten extends Layer {

    public Flatten() {}
    public Flatten(String name) { super(name); }

    @Override protected String defaultNamePrefix() { return "flatten"; }

    @Override public Shape computeOutputShape(Shape inputShape) {
        int[] d = inputShape.dimensions();
        if (d.length <= 1) return inputShape;
        int product = 1;
        for (int i = 1; i < d.length; i++) product *= d[i];
        return new Shape(d[0], product);
    }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {}

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        NDArray x = inputs.get(0);
        int[] d = x.shape().dimensions();
        if (d.length <= 1) return x;
        int product = 1;
        for (int i = 1; i < d.length; i++) product *= d[i];
        return x.reshape(new Shape(d[0], product));
    }
}
