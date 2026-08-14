package com.marmanis.jax4j.keras.optimizers;

import com.marmanis.jax4j.pytree.PyTree;

/**
 * A pure-functional optimizer over a {@link PyTree} of parameters. The
 * caller threads {@code state} between successive {@link #apply} calls; all
 * mutation is at the top-level training loop, keeping the tape clean.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Optimizer {

    /** Fresh optimizer state matching {@code params}' structure. */
    Object initState(PyTree params);

    /** Produce new params and next state from the current params, gradients, and state. */
    OptimStep apply(PyTree params, PyTree grads, Object state);

    /** Return of one optimization step. */
    record OptimStep(PyTree params, Object state) {}
}
