package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.ActivationFn;
import com.marmanis.jax4j.ml.nn.Linear;
import com.marmanis.jax4j.ml.nn.MLP;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModulesTest {

    @Test
    public void linearRoundTrip() {
        Linear layer = Linear.init(PRNGKey.key(1), 3, 4);
        PyTree tree = Modules.flatten(layer);
        assertTrue(tree instanceof PyTree.MapNode);
        Linear back = Modules.unflatten(layer, tree);
        assertArrayEquals(layer.weight().toFloatArray(), back.weight().toFloatArray());
        assertArrayEquals(layer.bias().toFloatArray(), back.bias().toFloatArray());
    }

    @Test
    public void mlpRoundTripPreservesStatic() {
        MLP mlp = MLP.init(PRNGKey.key(2), 4, 2, 8, 3, ActivationFn.TANH);
        PyTree tree = Modules.flatten(mlp);
        MLP back = Modules.unflatten(mlp, tree);
        assertSame(mlp.activation(), back.activation(), "@Static activation must be preserved");
        assertEquals(mlp.layers().size(), back.layers().size());
        for (int i = 0; i < mlp.layers().size(); i++) {
            assertArrayEquals(mlp.layers().get(i).weight().toFloatArray(),
                              back.layers().get(i).weight().toFloatArray());
        }
    }

    @Test
    public void treeMapAppliesToEveryLeaf() {
        Linear layer = Linear.init(PRNGKey.key(3), 2, 2);
        Linear doubled = Modules.treeMap(layer, a -> a.add(a));
        float[] orig = layer.weight().toFloatArray();
        float[] two = doubled.weight().toFloatArray();
        for (int i = 0; i < orig.length; i++) assertEquals(2f * orig[i], two[i], 1e-6);
    }

    @Test
    public void flattenExposesEveryParameterLeaf() {
        MLP mlp = MLP.init(PRNGKey.key(4), 5, 3, 6, 2, ActivationFn.RELU);
        List<NDArray> leaves = PyTrees.flatten(Modules.flatten(mlp));
        // 2 layers, each 2 leaves (w, b)
        assertEquals(4, leaves.size());
        for (NDArray l : leaves) assertNotNull(l);
    }

    @Test
    public void nestedSequentialInsideCustomRecord() {
        // Nested module inside a wrapper record — round-trips through Modules.
        Linear inner = new Linear(
            new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2)),
            new ConcreteNDArray(new float[]{0.5f, -0.5f}, new Shape(2)));
        Wrapper w = new Wrapper(inner, "tag");
        PyTree tree = Modules.flatten(w);
        Wrapper back = Modules.unflatten(w, tree);
        assertArrayEquals(inner.weight().toFloatArray(), back.inner().weight().toFloatArray());
        assertEquals("tag", back.name());
    }

    public record Wrapper(Linear inner, String name) implements Module {}
}
