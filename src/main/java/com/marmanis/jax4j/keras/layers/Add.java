package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.KerasTensor;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Elementwise addition of two or more upstream tensors — the tiny "merge"
 * layer used to compose skip / residual connections in the Functional API.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Add extends Layer {

    public Add() {}
    public Add(String name) { super(name); }

    @Override protected String defaultNamePrefix() { return "add"; }

    @Override public Shape computeOutputShape(Shape inputShape) { return inputShape; }

    @Override protected Shape computeOutputShape(List<KerasTensor> inputs) {
        return inputs.get(0).shape();
    }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {}

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        NDArray sum = inputs.get(0);
        for (int i = 1; i < inputs.size(); i++) sum = sum.add(inputs.get(i));
        return sum;
    }
}
