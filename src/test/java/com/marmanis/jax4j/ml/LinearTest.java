package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.Linear;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class LinearTest {

    @Test
    public void initHasCorrectShapes() {
        Linear l = Linear.init(PRNGKey.key(0), 5, 3);
        assertEquals(new Shape(5, 3), l.weight().shape());
        assertEquals(new Shape(3), l.bias().shape());
    }

    @Test
    public void forwardMatchesDotPlusBias() {
        NDArray w = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
        NDArray b = new ConcreteNDArray(new float[]{10f, 20f, 30f}, new Shape(3));
        Linear l = new Linear(w, b);
        NDArray x = new ConcreteNDArray(new float[]{1f, 1f, 2f, 2f}, new Shape(2, 2));
        NDArray y = l.apply(x);
        // Row0: [1,1] . W + b = [1+4, 2+5, 3+6] + [10,20,30] = [15,27,39]
        // Row1: [2,2] . W = [2+8, 4+10, 6+12] + b = [20,34,48]
        assertArrayEquals(new float[]{15f, 27f, 39f, 20f, 34f, 48f}, y.toFloatArray(), 1e-5f);
    }
}
