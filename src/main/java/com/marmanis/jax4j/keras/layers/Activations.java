package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * String-keyed activation lookup used by {@link Dense}, {@link Activation}, etc.
 * Values match Keras: {@code "relu"}, {@code "tanh"}, {@code "sigmoid"},
 * {@code "softmax"}, {@code "linear"}, {@code "gelu"}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Activations {
    private Activations() {}

    public static NDArray apply(String name, NDArray x) {
        if (name == null || name.equals("linear")) return x;
        return switch (name) {
            case "relu"    -> x.relu();
            case "tanh"    -> x.tanh();
            case "sigmoid" -> x.sigmoid();
            case "softmax" -> Nn.softmax(x);
            case "gelu"    -> gelu(x);
            default -> throw new IllegalArgumentException("Unknown activation: " + name);
        };
    }

    /**
     * Tanh-based GELU approximation (differentiable through jax4j primitives):
     * {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))}.
     */
    private static NDArray gelu(NDArray x) {
        NDArray half = new ConcreteNDArray(new float[]{0.5f}, new Shape(1));
        NDArray one = new ConcreteNDArray(new float[]{1f}, new Shape(1));
        NDArray c1 = new ConcreteNDArray(new float[]{(float) Math.sqrt(2.0 / Math.PI)}, new Shape(1));
        NDArray c2 = new ConcreteNDArray(new float[]{0.044715f}, new Shape(1));
        NDArray x3 = x.mul(x).mul(x);
        NDArray inner = x.add(x3.mul(c2)).mul(c1);
        return x.mul(half).mul(one.add(inner.tanh()));
    }
}
