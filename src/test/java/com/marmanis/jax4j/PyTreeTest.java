package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.scalar;
import static com.marmanis.jax4j.testutil.GradChecker.vector;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the {@link PyTree}/{@link PyTrees} flatten-unflatten-map utilities
 * and for {@code grad} over a pytree of parameters, modeled on the role of
 * upstream JAX's {@code tree_util_test.py} and the pytree-parameter cases in
 * {@code api_test.py}.
 */
public class PyTreeTest {

    @Test
    public void testFlattenListNode() {
        PyTree tree = PyTree.list(PyTree.leaf(vector(1, 2)), PyTree.leaf(vector(3, 4, 5)));
        List<NDArray> leaves = PyTrees.flatten(tree);
        assertEquals(2, leaves.size());
        assertEquals(2, leaves.get(0).shape().size());
        assertEquals(3, leaves.get(1).shape().size());
    }

    @Test
    public void testFlattenMapNode() {
        PyTree tree = PyTree.dictOf(Map.of("w", vector(1, 2), "b", scalar(3)));
        List<NDArray> leaves = PyTrees.flatten(tree);
        assertEquals(2, leaves.size());
    }

    @Test
    public void testUnflattenRoundTrip() {
        PyTree original = PyTree.list(PyTree.leaf(vector(1, 2)), PyTree.leaf(scalar(5)));
        List<NDArray> leaves = PyTrees.flatten(original);
        PyTree rebuilt = PyTrees.unflatten(original, leaves);
        assertEquals(leaves, PyTrees.flatten(rebuilt));
    }

    @Test
    public void testMapAppliesToEveryLeaf() {
        PyTree tree = PyTree.list(PyTree.leaf(vector(1, 2, 3)), PyTree.leaf(scalar(4)));
        PyTree doubled = PyTrees.map(x -> x.mul(scalar(2)), tree);

        List<NDArray> leaves = PyTrees.flatten(doubled);
        assertEquals(2.0f, leaves.get(0).toFloatArray()[0]);
        assertEquals(8.0f, leaves.get(1).toFloatArray()[0]);
    }

    @Test
    public void testMap2CombinesTwoTreesLeafwise() {
        PyTree params = PyTree.list(PyTree.leaf(vector(1, 2)), PyTree.leaf(scalar(10)));
        PyTree grads = PyTree.list(PyTree.leaf(vector(0.1f, 0.2f)), PyTree.leaf(scalar(1.0f)));

        PyTree updated = PyTrees.map2(NDArray::sub, params, grads);
        List<NDArray> leaves = PyTrees.flatten(updated);
        assertEquals(0.9f, leaves.get(0).toFloatArray()[0], 1e-5f);
        assertEquals(9.0f, leaves.get(1).toFloatArray()[0], 1e-5f);
    }

    @Test
    public void testGradTreeOverMapOfParams() {
        // loss(w, b) = sum((w*2 + b)^2) ; trivial quadratic with a known minimum direction
        Function<PyTree, NDArray> loss = params -> {
            List<NDArray> leaves = PyTrees.flatten(params);
            NDArray w = leaves.get(0);
            NDArray b = leaves.get(1);
            NDArray pred = w.mul(scalar(2)).add(b);
            return pred.mul(pred).sum();
        };

        // LinkedHashMap (not Map.of, whose iteration order is unspecified) so
        // flatten()'s leaf order is deterministically w then b.
        Map<String, PyTree> paramFields = new java.util.LinkedHashMap<>();
        paramFields.put("w", PyTree.leaf(vector(1.0f, -1.0f)));
        paramFields.put("b", PyTree.leaf(scalar(0.5f)));
        PyTree params = PyTree.dict(paramFields);

        PyTree grads = JAX.gradTree(loss).apply(params);
        List<NDArray> gradLeaves = PyTrees.flatten(grads);

        // d/dw sum((2w+b)^2) = 4*(2w+b) ; d/db sum((2w+b)^2) = 2*sum(2w+b)
        float[] w = {1.0f, -1.0f};
        float b = 0.5f;
        float[] expectedGradW = new float[w.length];
        float expectedGradB = 0;
        for (int i = 0; i < w.length; i++) {
            float pred = 2 * w[i] + b;
            expectedGradW[i] = 4 * pred;
            expectedGradB += 2 * pred;
        }

        assertEquals(expectedGradW[0], gradLeaves.get(0).toFloatArray()[0], 1e-4f);
        assertEquals(expectedGradW[1], gradLeaves.get(0).toFloatArray()[1], 1e-4f);
        assertEquals(expectedGradB, gradLeaves.get(1).toFloatArray()[0], 1e-4f);
    }

    @Test
    public void testUnflattenRejectsWrongLeafCount() {
        PyTree tree = PyTree.list(PyTree.leaf(vector(1, 2)), PyTree.leaf(scalar(3)));
        assertTrue(assertThrowsIllegalArgument(() -> PyTrees.unflatten(tree, List.of(vector(1, 2)))));
    }

    private static boolean assertThrowsIllegalArgument(Runnable r) {
        try {
            r.run();
            return false;
        } catch (IllegalArgumentException e) {
            return true;
        }
    }
}
