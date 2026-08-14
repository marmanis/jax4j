package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Shape;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Static factories for symbolic {@link KerasTensor} input placeholders, the
 * entry point into a functional-API {@link Model}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Input {
    private Input() {}

    private static final AtomicInteger COUNTER = new AtomicInteger();

    /** {@code input(shape)} with {@link DType#FLOAT32}. */
    public static KerasTensor input(Shape shape) {
        return input(shape, DType.FLOAT32, null);
    }

    public static KerasTensor input(Shape shape, DType dtype) {
        return input(shape, dtype, null);
    }

    public static KerasTensor input(Shape shape, DType dtype, String name) {
        String actual = (name != null) ? name : "input_" + COUNTER.getAndIncrement();
        return new KerasTensor(shape, dtype, new KerasTensor.InputNode(actual, shape, dtype));
    }
}
