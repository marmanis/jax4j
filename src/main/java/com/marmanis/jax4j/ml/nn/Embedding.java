package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;

/**
 * Learned embedding table of shape {@code [numEmbeddings, embeddingDim]}.
 * {@link #apply} performs a differentiable gather via {@link Numpy#take};
 * indices are INT32 of any shape and the output appends {@code embeddingDim}
 * as a trailing dimension.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Embedding(NDArray table) implements Module {

    public static Embedding init(PRNGKey key, int numEmbeddings, int embeddingDim) {
        NDArray t = Random.normal(key, new Shape(numEmbeddings, embeddingDim));
        return new Embedding(t);
    }

    public NDArray apply(NDArray indices) {
        return Numpy.take(table, indices);
    }
}
