package com.marmanis.jax4j.keras.losses;

import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * Static factories for the built-in Keras losses. Every returned {@link Loss}
 * is differentiable and self-contained.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Losses {
    private Losses() {}

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }

    private static NDArray scalarLike(float v, DType dtype) {
        if (dtype == DType.FLOAT64) {
            return new ConcreteNDArray(new double[]{v}, new Shape(1));
        }
        return scalar(v);
    }

    /** Mean of {@code (yPred - yTrue)^2}. */
    public static Loss mse() {
        return (yTrue, yPred) -> {
            NDArray diff = yPred.sub(yTrue);
            return diff.mul(diff).mean();
        };
    }

    /** Mean absolute error: {@code mean(|yPred - yTrue|)}, computed via {@code max(d, -d)}. */
    public static Loss mae() {
        return (yTrue, yPred) -> {
            NDArray diff = yPred.sub(yTrue);
            NDArray negOne = scalarLike(-1f, diff.dtype());
            NDArray neg = diff.mul(negOne);
            return diff.max(neg).mean();
        };
    }

    /** {@link #categoricalCrossentropy(boolean)} with {@code fromLogits=false}. */
    public static Loss categoricalCrossentropy() {
        return categoricalCrossentropy(false);
    }

    /**
     * Softmax cross-entropy. If {@code fromLogits}, {@code yPred} is unnormalized
     * logits and softmax is applied internally; otherwise {@code yPred} is treated
     * as a probability distribution over the last axis.
     */
    public static Loss categoricalCrossentropy(boolean fromLogits) {
        return (yTrue, yPred) -> {
            NDArray probs = fromLogits ? Nn.softmax(yPred) : yPred;
            NDArray eps = scalarLike(1e-7f, probs.dtype());
            NDArray logProbs = probs.add(eps).log();
            NDArray perExample = yTrue.mul(logProbs).sum(-1);
            NDArray neg = scalarLike(-1f, perExample.dtype());
            return perExample.mean().mul(neg);
        };
    }

    /** {@link #sparseCategoricalCrossentropy(boolean)} with {@code fromLogits=false}. */
    public static Loss sparseCategoricalCrossentropy() {
        return sparseCategoricalCrossentropy(false);
    }

    /**
     * Same as {@link #categoricalCrossentropy} but {@code yTrue} is {@code INT32}
     * class indices; a one-hot expansion is built internally.
     */
    public static Loss sparseCategoricalCrossentropy(boolean fromLogits) {
        Loss inner = categoricalCrossentropy(fromLogits);
        return (yTrue, yPred) -> {
            int[] dims = yPred.shape().dimensions();
            int numClasses = dims[dims.length - 1];
            int batch = 1;
            for (int i = 0; i < dims.length - 1; i++) batch *= dims[i];
            int[] labels = yTrue.toIntArray();
            if (yPred.dtype() == DType.FLOAT64) {
                double[] oh = new double[batch * numClasses];
                for (int i = 0; i < batch; i++) oh[i * numClasses + labels[i]] = 1.0;
                return inner.call(new ConcreteNDArray(oh, yPred.shape()), yPred);
            }
            float[] oh = new float[batch * numClasses];
            for (int i = 0; i < batch; i++) oh[i * numClasses + labels[i]] = 1f;
            return inner.call(new ConcreteNDArray(oh, yPred.shape()), yPred);
        };
    }

    /** {@link #binaryCrossentropy(boolean)} with {@code fromLogits=false}. */
    public static Loss binaryCrossentropy() {
        return binaryCrossentropy(false);
    }

    /** Binary cross-entropy: {@code -mean(y*log(p) + (1-y)*log(1-p))}. */
    public static Loss binaryCrossentropy(boolean fromLogits) {
        return (yTrue, yPred) -> {
            NDArray p = fromLogits ? yPred.sigmoid() : yPred;
            NDArray eps = scalarLike(1e-7f, p.dtype());
            NDArray one = scalarLike(1f, p.dtype());
            NDArray negOne = scalarLike(-1f, p.dtype());
            NDArray logP = p.add(eps).log();
            NDArray logOneMinusP = one.sub(p).add(eps).log();
            NDArray term = yTrue.mul(logP).add(one.sub(yTrue).mul(logOneMinusP));
            return term.mean().mul(negOne);
        };
    }
}
