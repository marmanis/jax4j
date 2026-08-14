package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Standalone activation layer applying a named activation function.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Activation extends Layer {
    private final String activation;

    public Activation(String activation) { this(activation, null); }
    public Activation(String activation, String name) { super(name); this.activation = activation; }

    @Override protected String defaultNamePrefix() { return "activation"; }

    @Override public Shape computeOutputShape(Shape inputShape) { return inputShape; }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {}

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        return Activations.apply(activation, inputs.get(0));
    }
}
