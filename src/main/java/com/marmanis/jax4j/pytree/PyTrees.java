package com.marmanis.jax4j.pytree;

import com.marmanis.jax4j.core.NDArray;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Flatten/unflatten/map utilities over {@link PyTree}, mirroring the role of
 * {@code jax.tree_util} (e.g. {@code tree_flatten}, {@code tree_unflatten},
 * {@code tree_map}) in upstream JAX.
 */
public final class PyTrees {
    private PyTrees() {}

    /**
     * Returns every {@link NDArray} leaf in {@code tree}, in a deterministic
     * depth-first order (list children in order, map children in insertion order).
     */
    public static List<NDArray> flatten(PyTree tree) {
        List<NDArray> leaves = new ArrayList<>();
        flattenInto(tree, leaves);
        return leaves;
    }

    private static void flattenInto(PyTree tree, List<NDArray> out) {
        switch (tree) {
            case PyTree.Leaf(var value)       -> out.add(value);
            case PyTree.ListNode(var children) -> { for (PyTree child : children) flattenInto(child, out); }
            case PyTree.MapNode(var childMap)  -> { for (PyTree child : childMap.values()) flattenInto(child, out); }
        }
    }

    /**
     * Rebuilds a tree with the same structure as {@code template}, but with
     * leaf values replaced by {@code leaves} (consumed in the same order
     * {@link #flatten} would produce). The number of leaves must match exactly.
     */
    public static PyTree unflatten(PyTree template, List<NDArray> leaves) {
        Iterator<NDArray> it = leaves.iterator();
        PyTree result = unflattenRec(template, it);
        if (it.hasNext()) {
            throw new IllegalArgumentException("Too many leaves for this tree structure");
        }
        return result;
    }

    private static PyTree unflattenRec(PyTree template, Iterator<NDArray> it) {
        return switch (template) {
            case PyTree.Leaf _ -> {
                if (!it.hasNext()) {
                    throw new IllegalArgumentException("Not enough leaves for this tree structure");
                }
                yield new PyTree.Leaf(it.next());
            }
            case PyTree.ListNode(var templateChildren) -> {
                List<PyTree> children = new ArrayList<>();
                for (PyTree child : templateChildren) children.add(unflattenRec(child, it));
                yield new PyTree.ListNode(children);
            }
            case PyTree.MapNode(var templateMap) -> {
                Map<String, PyTree> children = new LinkedHashMap<>();
                for (Map.Entry<String, PyTree> e : templateMap.entrySet()) {
                    children.put(e.getKey(), unflattenRec(e.getValue(), it));
                }
                yield new PyTree.MapNode(children);
            }
        };
    }

    /**
     * Applies {@code fn} to every leaf, preserving the tree's structure.
     * Mirrors {@code jax.tree_util.tree_map(fn, tree)}.
     */
    public static PyTree map(Function<NDArray, NDArray> fn, PyTree tree) {
        return switch (tree) {
            case PyTree.Leaf(var value)        -> new PyTree.Leaf(fn.apply(value));
            case PyTree.ListNode(var children)  -> new PyTree.ListNode(children.stream().map(c -> map(fn, c)).toList());
            case PyTree.MapNode(var childMap)   -> {
                Map<String, PyTree> result = new LinkedHashMap<>();
                childMap.forEach((k, v) -> result.put(k, map(fn, v)));
                yield new PyTree.MapNode(result);
            }
        };
    }

    /**
     * Applies a binary {@code fn} leafwise across two trees that must share
     * identical structure (e.g. an SGD update: {@code map2(NDArray::sub, params, scaledGrads)}).
     */
    public static PyTree map2(BiFunction<NDArray, NDArray, NDArray> fn, PyTree a, PyTree b) {
        List<NDArray> aLeaves = flatten(a);
        List<NDArray> bLeaves = flatten(b);
        if (aLeaves.size() != bLeaves.size()) {
            throw new IllegalArgumentException("Tree structures do not match: " + aLeaves.size() + " vs " + bLeaves.size() + " leaves");
        }
        List<NDArray> combined = new ArrayList<>();
        for (int i = 0; i < aLeaves.size(); i++) combined.add(fn.apply(aLeaves.get(i), bLeaves.get(i)));
        return unflatten(a, combined);
    }
}
