package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.optimizers.Adam;
import com.marmanis.jax4j.keras.optimizers.Optimizer;
import com.marmanis.jax4j.pytree.PyTree;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AdamTest {

    @Test
    public void oneStepUpdatesParams() {
        NDArray p = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
        NDArray g = new ConcreteNDArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2));
        PyTree params = PyTree.dict(java.util.Map.of("layer/kernel", PyTree.leaf(p)));
        PyTree grads = PyTree.dict(java.util.Map.of("layer/kernel", PyTree.leaf(g)));

        Adam adam = new Adam(1e-2f);
        Object state = adam.initState(params);
        Optimizer.OptimStep step = adam.apply(params, grads, state);
        NDArray newP = ((PyTree.Leaf) ((PyTree.MapNode) step.params()).children().get("layer/kernel")).value();

        float[] orig = p.toFloatArray();
        float[] now = newP.toFloatArray();
        for (int i = 0; i < orig.length; i++) {
            assertNotEquals(orig[i], now[i], "expected updated weight at " + i);
            assertTrue(Math.abs(orig[i] - now[i]) < 1f);
        }
        Adam.State s = (Adam.State) step.state();
        assertEquals(1, s.step());
    }
}
