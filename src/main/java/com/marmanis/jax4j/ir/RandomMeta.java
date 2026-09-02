package com.marmanis.jax4j.ir;

import com.marmanis.jax4j.core.Shape;

/**
 * Metadata for random primitives used in Jaxpr tracing.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public sealed interface RandomMeta {
    record Uniform(Shape shape, float lo, float hi) implements RandomMeta {}
    record Normal(Shape shape) implements RandomMeta {}
    record Bernoulli(float p, Shape shape) implements RandomMeta {}
    record Permutation(int n) implements RandomMeta {}
}
