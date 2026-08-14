package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.Shape;

import java.util.List;

/**
 * Straight-line stack of layers, the beginner's Keras API. Layers are held in
 * a plain list and re-wired into the parent {@link Model}'s DAG on every
 * {@link #add} — small overhead in exchange for a fluent API that stays in
 * sync as callers build the network up.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class Sequential extends Model {

    private final Shape inputShape;
    private KerasTensor currentTensor;

    /** Empty Sequential; call {@link #add} at least once with a known input shape. */
    public Sequential(Shape inputShape) {
        super();
        this.inputShape = inputShape;
        this.currentTensor = Input.input(inputShape);
        this.inputs.add(currentTensor);
    }

    /** Fluent factory: {@code Sequential.of(Shape.of(784))}. */
    public static Sequential of(Shape inputShape) {
        return new Sequential(inputShape);
    }

    /** Convenience: build from a pre-populated list. */
    public Sequential(Shape inputShape, List<Layer> layers) {
        this(inputShape);
        for (Layer l : layers) add(l);
    }

    /** Append a layer and rebuild the DAG so the model stays queryable. */
    public Sequential add(Layer layer) {
        currentTensor = layer.apply(currentTensor);
        outputs.clear();
        outputs.add(currentTensor);
        rebuild();
        return this;
    }

    /** Remove and return the most recently added layer. */
    public Layer removeLast() {
        if (layersInTopoOrder.isEmpty()) throw new IllegalStateException("no layers to remove");
        Layer last = layersInTopoOrder.get(layersInTopoOrder.size() - 1);
        // Rewind currentTensor to the previous layer's output (or the initial Input).
        if (layersInTopoOrder.size() == 1) {
            currentTensor = inputs.get(0);
        } else {
            KerasTensor.LayerNode prevNode = nodesInTopoOrder.get(nodesInTopoOrder.size() - 2);
            Layer prevLayer = prevNode.layer();
            currentTensor = new KerasTensor(
                prevLayer.computeOutputShape(prevNode.inputs().get(0).shape()),
                prevNode.inputs().get(0).dtype(),
                prevNode);
        }
        outputs.clear();
        outputs.add(currentTensor);
        rebuild();
        return last;
    }

    public Shape inputShape() { return inputShape; }
}
