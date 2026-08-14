package com.marmanis.jax4j.keras.optimizers;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

/**
 * Vanilla stochastic gradient descent: {@code p -= lr * g}. State is unused.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class SGD implements Optimizer {

    private final float lr;

    public SGD(float learningRate) { this.lr = learningRate; }

    @Override public Object initState(PyTree params) { return null; }

    @Override public OptimStep apply(PyTree params, PyTree grads, Object state) {
        NDArray lrArr32 = new ConcreteNDArray(new float[]{lr}, new Shape(1));
        NDArray lrArr64 = new ConcreteNDArray(new double[]{lr}, new Shape(1));
        PyTree newParams = PyTrees.map2((p, g) -> {
            NDArray scaled = g.mul(g.dtype() == DType.FLOAT64 ? lrArr64 : lrArr32);
            return p.sub(scaled);
        }, params, grads);
        return new OptimStep(newParams, null);
    }
}
