package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Dense;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DenseTest {

    @Test
    public void buildAllocatesKernelAndBiasWithCorrectShapes() {
        Dense d = new Dense(5);
        d.build(new Shape(3), new PRNGKey(0));
        assertEquals(new Shape(3, 5), d.getParams().get("kernel").shape());
        assertEquals(new Shape(5), d.getParams().get("bias").shape());
        assertTrue(d.isBuilt());
        assertEquals(3 * 5 + 5, d.countParams());
    }

    @Test
    public void forwardMatchesManualDotBias() {
        Dense d = new Dense(2);
        d.build(new Shape(3), new PRNGKey(1));

        NDArray x = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
        NDArray manual = x.dot(d.getParams().get("kernel")).add(d.getParams().get("bias"));
        NDArray via = d.call(x);
        float[] a = manual.toFloatArray();
        float[] b = via.toFloatArray();
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) assertEquals(a[i], b[i], 1e-6f);
    }
}
