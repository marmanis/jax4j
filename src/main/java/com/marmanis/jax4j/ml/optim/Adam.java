package com.marmanis.jax4j.ml.optim;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Modules;

/**
 * Bias-corrected Adam. State carries first- and second-moment estimates
 * ({@code m}, {@code v}) shaped like the parameter tree and the step counter.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Adam(float lr, float b1, float b2, float eps) implements Optimizer<Adam.State> {

    /** First moment, second moment, step counter (1-indexed after first update). */
    public record State(Module m, Module v, int step) {}

    public static Adam defaults(float lr) {
        return new Adam(lr, 0.9f, 0.999f, 1e-8f);
    }

    @Override
    public <M extends Module> State init(M module) {
        return new State(Modules.zerosLike(module), Modules.zerosLike(module), 0);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <M extends Module> Update<M, State> update(M module, M grads, State state) {
        int t = state.step + 1;
        M mPrev = (M) state.m;
        M vPrev = (M) state.v;
        M mNext = Modules.treeMap2(mPrev, grads, (mp, g) -> mp.mul(scalar(b1)).add(g.mul(scalar(1f - b1))));
        M vNext = Modules.treeMap2(vPrev, grads, (vp, g) -> vp.mul(scalar(b2)).add(g.mul(g).mul(scalar(1f - b2))));

        float bc1 = (float) (1.0 - Math.pow(b1, t));
        float bc2 = (float) (1.0 - Math.pow(b2, t));
        M update = adamUpdateTree(mNext, vNext, bc1, bc2, eps, lr);
        M newModule = Modules.treeMap2(module, update, NDArray::sub);
        return new Update<>(newModule, new State(mNext, vNext, t));
    }

    /**
     * Combines the two moment trees into {@code lr * mHat / (sqrt(vHat) + eps)}
     * per leaf. Element-wise math via raw float arrays (both trees are always
     * concrete here — this runs outside any gradient tape).
     */
    static <M extends Module> M adamUpdateTree(M mNext, M vNext, float bc1, float bc2, float eps, float lr) {
        return Modules.treeMap2(mNext, vNext, (m, v) -> {
            float[] mv = m.toFloatArray();
            float[] vv = v.toFloatArray();
            float[] out = new float[mv.length];
            for (int i = 0; i < mv.length; i++) {
                float mHat = mv[i] / bc1;
                float vHat = vv[i] / bc2;
                out[i] = lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
            return new ConcreteNDArray(out, m.shape());
        });
    }

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }
}
