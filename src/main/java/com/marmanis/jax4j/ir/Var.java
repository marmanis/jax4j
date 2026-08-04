package com.marmanis.jax4j.ir;

import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.core.DType;

/**
 * A variable in a Jaxpr.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record Var(int id, Shape shape, DType dtype) {
    @Override
    public String toString() {
        return "v" + id + shape;
    }
}
