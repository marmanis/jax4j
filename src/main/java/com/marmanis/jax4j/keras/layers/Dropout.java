package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Inverted dropout: at train time zeroes each element independently with
 * probability {@code rate} and rescales survivors by {@code 1/(1-rate)}. A
 * no-op during evaluation.
 *
 * <p>Uses a self-derived {@link PRNGKey} seeded from the current nanos xor'd
 * with the layer name — good enough for training-time stochasticity, since
 * jax4j's core RNG is otherwise purely functional and we don't want to plumb
 * a global {@code PRNGKey} through every layer's {@code call}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Dropout extends Layer {
    private final float rate;
    private long callCounter = 0;

    public Dropout(float rate) { this(rate, null); }
    public Dropout(float rate, String name) { super(name); this.rate = rate; }

    @Override protected String defaultNamePrefix() { return "dropout"; }

    @Override public Shape computeOutputShape(Shape inputShape) { return inputShape; }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {}

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        NDArray x = inputs.get(0);
        if (!training || rate <= 0f) return x;
        long seed = (System.nanoTime() ^ name.hashCode() ^ (++callCounter * 0x9E3779B97F4A7C15L));
        return Nn.dropout(x, new PRNGKey(seed), rate, true);
    }
}
