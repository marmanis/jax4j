package com.marmanis.jax4j.core;

import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.List;

/**
 * An explicit, immutable pseudo-random key, mirroring {@code jax.random.PRNGKey}.
 * jax4j threads randomness explicitly through {@link com.marmanis.jax4j.api.Random}
 * rather than relying on hidden global RNG state: every sampling call takes a key
 * and is a pure function of it, and {@link com.marmanis.jax4j.api.Random#split}
 * derives independent child keys for use in different parts of a computation
 * (e.g. one subkey per layer's weight init, another for a dropout mask).
 *
 * <p>Now backed by an {@link NDArray} so it can be dynamically traced in a {@code Jaxpr}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record PRNGKey(NDArray keyArray) {

    public PRNGKey(long state) {
        this(new ConcreteNDArray(new long[]{state}, new Shape()));
    }

    public long state() {
        return keyArray.toLongArray()[0];
    }

    /** Mirrors {@code jax.random.PRNGKey(seed)} / {@code jax.random.key(seed)}. */
    public static PRNGKey key(long seed) {
        if (Tracer.current() != null) {
            Tracer tracer = Tracer.current();
            Var outVar = tracer.nextVar(new Shape(), DType.INT64);
            tracer.addEquation(new Equation(
                List.of(),
                List.of(outVar),
                Primitive.RANDOM_SEED,
                seed
            ));
            return new PRNGKey(new TracedNDArray(outVar));
        } else {
            return new PRNGKey(seed);
        }
    }
}

