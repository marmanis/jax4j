package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.Linear;
import com.marmanis.jax4j.ml.optim.Adam;
import com.marmanis.jax4j.ml.optim.Optimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AdamTest {

    @Test
    public void oneStepIsBoundedAndNonZero() {
        NDArray w = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
        NDArray b = new ConcreteNDArray(new float[]{0f, 0f}, new Shape(2));
        Linear layer = new Linear(w, b);
        Linear grads = new Linear(
            new ConcreteNDArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2)),
            new ConcreteNDArray(new float[]{0f, 0f}, new Shape(2)));

        Adam adam = Adam.defaults(1e-2f);
        Optimizer.Update<Linear, Adam.State> upd = adam.update(layer, grads, adam.init(layer));

        float[] out = upd.newModule().weight().toFloatArray();
        float[] orig = w.toFloatArray();
        for (int i = 0; i < out.length; i++) {
            float delta = orig[i] - out[i];
            assertNotEquals(0f, delta, "expected nonzero update on weight " + i);
            assertTrue(Math.abs(delta) < 1f, "unexpectedly large step: " + delta);
        }
        assertEquals(w.shape(), upd.newModule().weight().shape());
        assertEquals(1, upd.newState().step());
    }
}
