package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * A pointwise activation, held as a value so a module can carry it as a
 * {@link com.marmanis.jax4j.ml.Static @Static} hyperparameter. Constants
 * {@link #RELU}, {@link #TANH}, {@link #SIGMOID}, {@link #GELU} cover the
 * usual set; custom activations are just lambdas.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
@FunctionalInterface
public interface ActivationFn {
    NDArray apply(NDArray x);

    ActivationFn RELU = NDArray::relu;
    ActivationFn TANH = NDArray::tanh;
    ActivationFn SIGMOID = NDArray::sigmoid;

    /**
     * Approximate GELU: {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))}.
     */
    ActivationFn GELU = x -> {
        NDArray half = new ConcreteNDArray(new float[]{0.5f}, new Shape(1));
        NDArray one = new ConcreteNDArray(new float[]{1f}, new Shape(1));
        NDArray c1 = new ConcreteNDArray(new float[]{(float) Math.sqrt(2.0 / Math.PI)}, new Shape(1));
        NDArray c2 = new ConcreteNDArray(new float[]{0.044715f}, new Shape(1));
        NDArray x3 = x.mul(x).mul(x);
        NDArray inner = x.add(x3.mul(c2)).mul(c1);
        return half.mul(x).mul(one.add(inner.tanh()));
    };
}
