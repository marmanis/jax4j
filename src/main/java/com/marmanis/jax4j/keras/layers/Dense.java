package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Standard fully-connected layer: {@code y = activation(x . kernel + bias)}.
 * Kernel is Glorot-uniform initialized; bias starts at zero.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Dense extends Layer {

    private final int units;
    private final String activation;
    private final boolean useBias;

    public Dense(int units) { this(units, null, true, null); }
    public Dense(int units, String activation) { this(units, activation, true, null); }
    public Dense(int units, String activation, boolean useBias, String name) {
        super(name);
        this.units = units;
        this.activation = activation;
        this.useBias = useBias;
    }

    @Override protected String defaultNamePrefix() { return "dense"; }

    public int units() { return units; }
    public String activation() { return activation; }

    @Override public Shape computeOutputShape(Shape inputShape) {
        int[] dims = inputShape.dimensions();
        int[] out = dims.clone();
        out[out.length - 1] = units;
        return new Shape(out);
    }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {
        int[] dims = inputShape.dimensions();
        int fanIn = dims[dims.length - 1];
        PRNGKey[] sub = Random.split(key, 2);
        NDArray kernel = Random.glorotUniform(sub[0], new Shape(fanIn, units), fanIn, units);
        paramNames.add("kernel");
        params.put("kernel", kernel);
        if (useBias) {
            NDArray bias = new ConcreteNDArray(new float[units], new Shape(units));
            paramNames.add("bias");
            params.put("bias", bias);
        }
    }

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        NDArray x = inputs.get(0);
        NDArray kernel = effectiveParams.get("kernel");
        NDArray y = x.dot(kernel);
        if (useBias) y = y.add(effectiveParams.get("bias"));
        return Activations.apply(activation, y);
    }
}
