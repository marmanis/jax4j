package com.marmanis.jax4j.ml.optim;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Modules;

/**
 * Adam with decoupled weight decay (Loshchilov & Hutter, 2019): after the
 * regular Adam step, subtract {@code lr * weightDecay * p} from every
 * parameter. Uses {@link Adam.State} unchanged.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record AdamW(float lr, float b1, float b2, float eps, float weightDecay) implements Optimizer<Adam.State> {

    public static AdamW defaults(float lr, float weightDecay) {
        return new AdamW(lr, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    @Override
    public <M extends Module> Adam.State init(M module) {
        return new Adam(lr, b1, b2, eps).init(module);
    }

    @Override
    public <M extends Module> Update<M, Adam.State> update(M module, M grads, Adam.State state) {
        Update<M, Adam.State> adamStep = new Adam(lr, b1, b2, eps).update(module, grads, state);
        NDArray decay = new ConcreteNDArray(new float[]{lr * weightDecay}, new Shape(1));
        M decayed = Modules.treeMap2(adamStep.newModule(), module, (pNew, pOld) -> pNew.sub(pOld.mul(decay)));
        return new Update<>(decayed, adamStep.newState());
    }
}
