package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.ActivationFn;
import com.marmanis.jax4j.ml.nn.MLP;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MLPTest {

    @Test
    public void forwardShapeMatches() {
        MLP mlp = MLP.init(PRNGKey.key(7), 8, 4, 6, 3, ActivationFn.RELU);
        NDArray x = new com.marmanis.jax4j.core.ConcreteNDArray(new float[16], new Shape(2, 8));
        NDArray y = mlp.apply(x);
        assertEquals(new Shape(2, 4), y.shape());
    }

    @Test
    public void parameterCountIsSumOfLayerParams() {
        MLP mlp = MLP.init(PRNGKey.key(8), 8, 4, 6, 3, ActivationFn.RELU);
        List<NDArray> leaves = PyTrees.flatten(Modules.flatten(mlp));
        // depth=3 layers, 2 leaves each (w, b)
        assertEquals(6, leaves.size());
        long total = 0;
        for (NDArray l : leaves) total += l.shape().size();
        // L0: 8*6 + 6 ; L1: 6*6 + 6 ; L2: 6*4 + 4
        assertEquals((8*6 + 6) + (6*6 + 6) + (6*4 + 4), total);
    }
}
