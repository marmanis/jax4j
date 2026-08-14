package com.marmanis.jax4j.ml.optim;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Modules;

/**
 * Plain stochastic gradient descent: {@code p <- p - lr * g}. Stateless.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record SGD(float lr) implements Optimizer<SGD.Unit> {

    /** Empty state marker. */
    public record Unit() {}

    @Override
    public <M extends Module> Unit init(M module) {
        return new Unit();
    }

    @Override
    public <M extends Module> Update<M, Unit> update(M module, M grads, Unit state) {
        NDArray lrArr = new ConcreteNDArray(new float[]{lr}, new Shape(1));
        M newModule = Modules.treeMap2(module, grads, (p, g) -> p.sub(g.mul(lrArr)));
        return new Update<>(newModule, state);
    }
}
