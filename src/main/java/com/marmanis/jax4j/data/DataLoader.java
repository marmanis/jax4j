package com.marmanis.jax4j.data;

import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * An iterable data loader that yields batches of examples.
 * Supports random shuffling via an explicit {@link PRNGKey} and collates
 * leaf elements from individual PyTrees into batched PyTrees.
 */
public class DataLoader implements Iterable<PyTree> {
    private final Dataset dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final PRNGKey key;

    /**
     * Creates a new DataLoader.
     *
     * @param dataset   The dataset to load from.
     * @param batchSize The size of each batch.
     * @param shuffle   Whether to shuffle the dataset index at the beginning of each iteration.
     * @param key       The random key used for shuffling (can be null if shuffle is false).
     */
    public DataLoader(Dataset dataset, int batchSize, boolean shuffle, PRNGKey key) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.key = key;
    }

    @Override
    public Iterator<PyTree> iterator() {
        int size = dataset.size();
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        if (shuffle && key != null) {
            NDArray rand = Random.uniform(key, new Shape(size));
            float[] randVals = rand.toFloatArray();
            indices.sort((a, b) -> Float.compare(randVals[a], randVals[b]));
        }

        return new Iterator<PyTree>() {
            private int cursor = 0;

            @Override
            public boolean hasNext() {
                return cursor < size;
            }

            @Override
            public PyTree next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                int end = Math.min(cursor + batchSize, size);
                List<PyTree> batchItems = new ArrayList<>(end - cursor);
                for (int i = cursor; i < end; i++) {
                    batchItems.add(dataset.get(indices.get(i)));
                }
                cursor = end;
                return collate(batchItems);
            }
        };
    }

    /**
     * Recursively collates a list of example PyTrees into a single batched PyTree.
     * Leaf arrays are stacked along a new leading batch dimension.
     */
    private static PyTree collate(List<PyTree> trees) {
        if (trees.isEmpty()) {
            throw new IllegalArgumentException("Cannot collate an empty list of trees");
        }
        PyTree first = trees.get(0);
        if (first instanceof PyTree.Leaf) {
            List<NDArray> leaves = new ArrayList<>(trees.size());
            for (PyTree t : trees) {
                leaves.add(((PyTree.Leaf) t).value());
            }
            return PyTree.leaf(stack(leaves));
        } else if (first instanceof PyTree.ListNode) {
            int numChildren = ((PyTree.ListNode) first).children().size();
            List<PyTree> collatedChildren = new ArrayList<>(numChildren);
            for (int i = 0; i < numChildren; i++) {
                List<PyTree> childTrees = new ArrayList<>(trees.size());
                for (PyTree t : trees) {
                    childTrees.add(((PyTree.ListNode) t).children().get(i));
                }
                collatedChildren.add(collate(childTrees));
            }
            return new PyTree.ListNode(collatedChildren);
        } else if (first instanceof PyTree.MapNode) {
            java.util.Set<String> keys = ((PyTree.MapNode) first).children().keySet();
            java.util.Map<String, PyTree> collatedChildren = new java.util.LinkedHashMap<>();
            for (String key : keys) {
                List<PyTree> childTrees = new ArrayList<>(trees.size());
                for (PyTree t : trees) {
                    childTrees.add(((PyTree.MapNode) t).children().get(key));
                }
                collatedChildren.put(key, collate(childTrees));
            }
            return new PyTree.MapNode(collatedChildren);
        } else {
            throw new IllegalArgumentException("Unsupported PyTree node type: " + first.getClass());
        }
    }

    /** Stacks a list of NDArrays along a new leading dimension. */
    private static NDArray stack(List<NDArray> arrays) {
        if (arrays.isEmpty()) {
            throw new IllegalArgumentException("Cannot stack empty list");
        }
        Shape singleShape = arrays.get(0).shape();
        int[] singleDims = singleShape.dimensions();
        int[] targetDims = new int[singleDims.length + 1];
        targetDims[0] = 1;
        System.arraycopy(singleDims, 0, targetDims, 1, singleDims.length);
        Shape reshapedShape = new Shape(targetDims);

        List<NDArray> reshaped = new ArrayList<>(arrays.size());
        for (NDArray a : arrays) {
            reshaped.add(a.reshape(reshapedShape));
        }
        return Numpy.concatenate(reshaped, 0);
    }
}
