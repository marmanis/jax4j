package com.marmanis.jax4j.keras.layers;

import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.Layer;

import java.util.List;
import java.util.Map;

/**
 * Embedding table lookup: maps {@code INT32} token ids of shape {@code [...]}
 * to embeddings of shape {@code [..., embeddingDim]}. Backed by
 * {@link Numpy#take} so gradients accumulate correctly into repeated rows.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Embedding extends Layer {

    private final int vocabSize;
    private final int embeddingDim;

    public Embedding(int vocabSize, int embeddingDim) { this(vocabSize, embeddingDim, null); }
    public Embedding(int vocabSize, int embeddingDim, String name) {
        super(name);
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
    }

    @Override protected String defaultNamePrefix() { return "embedding"; }

    @Override public Shape computeOutputShape(Shape inputShape) {
        int[] d = inputShape.dimensions();
        int[] out = new int[d.length + 1];
        System.arraycopy(d, 0, out, 0, d.length);
        out[d.length] = embeddingDim;
        return new Shape(out);
    }

    @Override protected void doBuild(Shape inputShape, PRNGKey key) {
        NDArray table = Random.normal(key, new Shape(vocabSize, embeddingDim));
        NDArray scale = new ConcreteNDArray(new float[]{0.01f}, new Shape(1));
        params.put("embeddings", table.mul(scale));
        paramNames.add("embeddings");
    }

    @Override protected NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training) {
        return Numpy.take(effectiveParams.get("embeddings"), inputs.get(0));
    }
}
