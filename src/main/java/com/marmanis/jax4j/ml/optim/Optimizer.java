package com.marmanis.jax4j.ml.optim;

import com.marmanis.jax4j.ml.Module;

/**
 * Purely functional optimizer: {@link #init} builds initial state for a
 * module, {@link #update} takes a module, its gradients, and current state,
 * returning a new module and new state.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Optimizer<S> {

    <M extends Module> S init(M module);

    <M extends Module> Update<M, S> update(M module, M grads, S state);

    /** Result of one optimizer step. */
    record Update<M extends Module, S>(M newModule, S newState) {}
}
