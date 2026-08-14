package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Add;

import java.util.List;

/**
 * Symbolic tensor used to describe a Keras Functional API model. Carries a
 * shape and dtype but no data; every {@link Layer#apply(KerasTensor)} call
 * records a new node in the DAG and returns a new {@code KerasTensor} whose
 * source is the layer node.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record KerasTensor(Shape shape, DType dtype, KerasNode source) {

    /** A node in the Keras functional-API graph. */
    public sealed interface KerasNode permits InputNode, LayerNode {}

    /** A named input placeholder — the entry point into a model. */
    public record InputNode(String name, Shape shape, DType dtype) implements KerasNode {}

    /** A layer applied to one or more upstream tensors. */
    public record LayerNode(Layer layer, List<KerasTensor> inputs) implements KerasNode {}

    /** Convenience: {@code a.add(b)} inserts a {@code layers.Add} node into the graph. */
    public KerasTensor add(KerasTensor other) {
        return new Add().apply(this, other);
    }
}
