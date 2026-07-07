package com.marmanis.jax4j.pytree;

import com.marmanis.jax4j.core.NDArray;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A recursive container of {@link NDArray} leaves, mirroring the role of
 * "pytrees" in upstream JAX: nested lists and (string-keyed) dicts of
 * arrays that {@code grad} and other transformations can flatten, transform
 * leafwise, and rebuild with the same structure.
 *
 * <p>This is intentionally a small, closed hierarchy (no custom-registered
 * node types) covering the common cases: a single array, an ordered list of
 * sub-trees (e.g. positional parameters), or a named map of sub-trees (e.g.
 * {@code {"w": ..., "b": ...}} model parameters).
 */
public sealed interface PyTree permits PyTree.Leaf, PyTree.ListNode, PyTree.MapNode {

    record Leaf(NDArray value) implements PyTree {}

    record ListNode(List<PyTree> children) implements PyTree {}

    record MapNode(Map<String, PyTree> children) implements PyTree {}

    static PyTree leaf(NDArray value) {
        return new Leaf(value);
    }

    static PyTree list(PyTree... children) {
        return new ListNode(List.of(children));
    }

    static PyTree dict(Map<String, PyTree> children) {
        return new MapNode(new LinkedHashMap<>(children));
    }

    /**
     * Convenience constructor for the common case of a flat dict of arrays,
     * e.g. {@code PyTree.dictOf(Map.of("w", w, "b", b))}.
     */
    static PyTree dictOf(Map<String, NDArray> arrays) {
        Map<String, PyTree> children = new LinkedHashMap<>();
        arrays.forEach((k, v) -> children.put(k, leaf(v)));
        return new MapNode(children);
    }
}
