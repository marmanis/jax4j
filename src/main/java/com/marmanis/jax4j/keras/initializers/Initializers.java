package com.marmanis.jax4j.keras.initializers;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;

/**
 * Common Keras-style parameter initializers. All are pure functions of a
 * {@link PRNGKey} and a shape.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Initializers {
    private Initializers() {}

    public static NDArray zeros(Shape shape) {
        return new ConcreteNDArray(new float[(int) shape.size()], shape);
    }

    public static NDArray ones(Shape shape) {
        float[] d = new float[(int) shape.size()];
        java.util.Arrays.fill(d, 1f);
        return new ConcreteNDArray(d, shape);
    }

    /** Standard normal samples scaled by {@code stddev}. */
    public static NDArray normal(PRNGKey key, Shape shape, float stddev) {
        NDArray n = Random.normal(key, shape);
        if (stddev == 1f) return n;
        return n.mul(new ConcreteNDArray(new float[]{stddev}, new Shape(1)));
    }

    /** Glorot/Xavier uniform: {@code U(-limit, limit)} with {@code limit=sqrt(6/(fanIn+fanOut))}. */
    public static NDArray glorotUniform(PRNGKey key, Shape shape, int fanIn, int fanOut) {
        return Random.glorotUniform(key, shape, fanIn, fanOut);
    }

    /** He/Kaiming normal: {@code N(0, sqrt(2/fanIn))}. */
    public static NDArray heNormal(PRNGKey key, Shape shape, int fanIn) {
        return Random.heNormal(key, shape, fanIn);
    }
}
