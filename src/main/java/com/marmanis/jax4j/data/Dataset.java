package com.marmanis.jax4j.data;

import com.marmanis.jax4j.pytree.PyTree;

/**
 * Represents a collection of examples that can be retrieved by index.
 * Useful for loading, batching, and training on datasets.
 */
public interface Dataset {

    /** Returns the total number of examples in the dataset. */
    int size();

    /**
     * Retrieves the example at the specified index.
     * The returned example is a PyTree (often a list of input and target leaf arrays).
     */
    PyTree get(int index);
}
