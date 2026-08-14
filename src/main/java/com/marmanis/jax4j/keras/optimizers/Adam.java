package com.marmanis.jax4j.keras.optimizers;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

/**
 * Bias-corrected Adam. State carries first ({@code m}) and second ({@code v})
 * moment PyTrees plus the step counter.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Adam implements Optimizer {

    private final float lr, b1, b2, eps;

    public Adam(float learningRate) { this(learningRate, 0.9f, 0.999f, 1e-8f); }

    public Adam(float learningRate, float beta1, float beta2, float epsilon) {
        this.lr = learningRate; this.b1 = beta1; this.b2 = beta2; this.eps = epsilon;
    }

    public record State(PyTree m, PyTree v, int step) {}

    @Override public Object initState(PyTree params) {
        PyTree z = PyTrees.map(p -> new ConcreteNDArray(new float[(int) p.shape().size()], p.shape()), params);
        return new State(z, z, 0);
    }

    @Override public OptimStep apply(PyTree params, PyTree grads, Object stateObj) {
        State state = (State) stateObj;
        int t = state.step + 1;
        float bc1 = (float) (1.0 - Math.pow(b1, t));
        float bc2 = (float) (1.0 - Math.pow(b2, t));
        PyTree mNext = PyTrees.map2((mp, g) -> lerp(mp, g, b1), state.m, grads);
        PyTree vNext = PyTrees.map2((vp, g) -> lerp(vp, g.mul(g), b2), state.v, grads);
        PyTree updates = PyTrees.map2((m, v) -> adamStep(m, v, bc1, bc2), mNext, vNext);
        PyTree newParams = PyTrees.map2(NDArray::sub, params, updates);
        return new OptimStep(newParams, new State(mNext, vNext, t));
    }

    private NDArray lerp(NDArray oldVal, NDArray newVal, float beta) {
        float[] a = oldVal.toFloatArray();
        float[] b = newVal.toFloatArray();
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = beta * a[i] + (1f - beta) * b[i];
        return new ConcreteNDArray(out, oldVal.shape());
    }

    private NDArray adamStep(NDArray m, NDArray v, float bc1, float bc2) {
        float[] mv = m.toFloatArray();
        float[] vv = v.toFloatArray();
        float[] out = new float[mv.length];
        for (int i = 0; i < mv.length; i++) {
            float mHat = mv[i] / bc1;
            float vHat = vv[i] / bc2;
            out[i] = lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
        return new ConcreteNDArray(out, m.shape());
    }
}
