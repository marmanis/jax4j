package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NnTest {

    @Test
    public void softmaxRowsSumToOne() {
        NDArray x = new ConcreteNDArray(new float[]{1f, 2f, 3f, 1f, 1f, 1f}, new Shape(2, 3));
        float[] result = Nn.softmax(x).toFloatArray();

        assertEquals(1f, result[0] + result[1] + result[2], 1e-5);
        assertEquals(1f, result[3] + result[4] + result[5], 1e-5);
        // uniform logits -> uniform softmax
        assertEquals(1f / 3f, result[3], 1e-5);
        assertEquals(1f / 3f, result[4], 1e-5);
        assertEquals(1f / 3f, result[5], 1e-5);
    }

    @Test
    public void softmaxMatchesKnownValues() {
        // softmax([0, log(2)]) = [1/3, 2/3]
        NDArray x = new ConcreteNDArray(new float[]{0f, (float) Math.log(2)}, new Shape(1, 2));
        float[] result = Nn.softmax(x).toFloatArray();
        assertEquals(1f / 3f, result[0], 1e-4);
        assertEquals(2f / 3f, result[1], 1e-4);
    }

    @Test
    public void softmaxGradientOfSumIsZero() {
        // sum(softmax(x)) is always 1 regardless of x, so its gradient is 0 everywhere.
        Function<NDArray, NDArray> fn = x -> Nn.softmax(x).sum();
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = new ConcreteNDArray(new float[]{0.5f, -1.2f, 3f}, new Shape(1, 3));
        float[] grad = gradFn.apply(x).toFloatArray();
        for (float g : grad) {
            assertEquals(0f, g, 1e-3);
        }
    }
}
