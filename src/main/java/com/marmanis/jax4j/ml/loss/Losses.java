package com.marmanis.jax4j.ml.loss;

import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * Standard scalar losses; every function is composed from differentiable
 * primitives so it works transparently under {@code JAX.grad}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Losses {
    private Losses() {}

    /** Mean of {@code (pred - target)^2}. */
    public static NDArray mse(NDArray pred, NDArray target) {
        NDArray diff = pred.sub(target);
        return diff.mul(diff).mean();
    }

    /** {@code -sum(target * log(softmax(logits)), axis=-1).mean()}. */
    public static NDArray softmaxCrossEntropy(NDArray logits, NDArray oneHotTargets) {
        NDArray logProbs = Nn.softmax(logits).log();
        NDArray perExample = oneHotTargets.mul(logProbs).sum(-1);
        NDArray neg = new ConcreteNDArray(new float[]{-1f}, new Shape(1));
        return perExample.mean().mul(neg);
    }

    /**
     * Same as {@link #softmaxCrossEntropy} but takes INT32 label indices; a
     * one-hot expansion is built internally so this is a drop-in for the
     * common "labels are class indices" case.
     */
    public static NDArray sparseCrossEntropy(NDArray logits, NDArray labelIndicesInt32) {
        int[] dims = logits.shape().dimensions();
        int numClasses = dims[dims.length - 1];
        int batch = 1;
        for (int i = 0; i < dims.length - 1; i++) batch *= dims[i];
        int[] labels = labelIndicesInt32.toIntArray();
        float[] oneHot = new float[batch * numClasses];
        for (int i = 0; i < batch; i++) oneHot[i * numClasses + labels[i]] = 1f;
        NDArray oh = new ConcreteNDArray(oneHot, logits.shape()).astype(logits.dtype() == DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32);
        return softmaxCrossEntropy(logits, oh);
    }
}
