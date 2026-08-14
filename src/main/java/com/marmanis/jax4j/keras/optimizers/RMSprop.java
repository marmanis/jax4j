package com.marmanis.jax4j.keras.optimizers;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

/**
 * RMSprop: moving average of squared gradients, {@code p -= lr * g / (sqrt(v) + eps)}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class RMSprop implements Optimizer {

    private final float lr, rho, eps;

    public RMSprop(float learningRate) { this(learningRate, 0.9f, 1e-7f); }

    public RMSprop(float learningRate, float rho, float epsilon) {
        this.lr = learningRate; this.rho = rho; this.eps = epsilon;
    }

    public record State(PyTree v) {}

    @Override public Object initState(PyTree params) {
        PyTree z = PyTrees.map(p -> new ConcreteNDArray(new float[(int) p.shape().size()], p.shape()), params);
        return new State(z);
    }

    @Override public OptimStep apply(PyTree params, PyTree grads, Object stateObj) {
        State state = (State) stateObj;
        PyTree vNext = PyTrees.map2((vp, g) -> lerp(vp, g.mul(g)), state.v, grads);
        PyTree updates = PyTrees.map2(this::step, grads, vNext);
        PyTree newParams = PyTrees.map2(NDArray::sub, params, updates);
        return new OptimStep(newParams, new State(vNext));
    }

    private NDArray lerp(NDArray oldVal, NDArray newVal) {
        float[] a = oldVal.toFloatArray();
        float[] b = newVal.toFloatArray();
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = rho * a[i] + (1f - rho) * b[i];
        return new ConcreteNDArray(out, oldVal.shape());
    }

    private NDArray step(NDArray g, NDArray v) {
        float[] gv = g.toFloatArray();
        float[] vv = v.toFloatArray();
        float[] out = new float[gv.length];
        for (int i = 0; i < gv.length; i++) {
            out[i] = lr * gv[i] / ((float) Math.sqrt(vv[i]) + eps);
        }
        return new ConcreteNDArray(out, g.shape());
    }
}
