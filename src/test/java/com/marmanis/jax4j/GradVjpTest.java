package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GradVjpTest {

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }

    @Test
    public void testDivGrad() {
        // f(x) = 10 / x ; f'(x) = -10 / x^2
        Function<NDArray, NDArray> fn = x -> scalar(10f).div(x);
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(2f);
        float result = gradFn.apply(x).toFloatArray()[0];
        assertEquals(-10f / (2f * 2f), result, 1e-4);
    }

    @Test
    public void testExpGrad() {
        // f(x) = exp(x) ; f'(x) = exp(x)
        Function<NDArray, NDArray> fn = NDArray::exp;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(1.5f);
        float result = gradFn.apply(x).toFloatArray()[0];
        assertEquals((float) Math.exp(1.5), result, 1e-4);
    }

    @Test
    public void testLogGrad() {
        // f(x) = log(x) ; f'(x) = 1/x
        Function<NDArray, NDArray> fn = NDArray::log;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(4f);
        float result = gradFn.apply(x).toFloatArray()[0];
        assertEquals(1f / 4f, result, 1e-4);
    }

    @Test
    public void testSinGrad() {
        // f(x) = sin(x) ; f'(x) = cos(x)
        Function<NDArray, NDArray> fn = NDArray::sin;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(0.3f);
        float result = gradFn.apply(x).toFloatArray()[0];
        assertEquals((float) Math.cos(0.3), result, 1e-4);
    }

    @Test
    public void testCosGrad() {
        // f(x) = cos(x) ; f'(x) = -sin(x)
        Function<NDArray, NDArray> fn = NDArray::cos;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(0.3f);
        float result = gradFn.apply(x).toFloatArray()[0];
        assertEquals((float) -Math.sin(0.3), result, 1e-4);
    }

    @Test
    public void testTanhGrad() {
        // f(x) = tanh(x) ; f'(x) = 1 - tanh(x)^2
        Function<NDArray, NDArray> fn = NDArray::tanh;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(0.6f);
        float result = gradFn.apply(x).toFloatArray()[0];
        float t = (float) Math.tanh(0.6);
        assertEquals(1f - t * t, result, 1e-4);
    }

    @Test
    public void testReluGrad() {
        // f(x) = relu(x) ; f'(x) = 1 for x > 0, 0 for x < 0
        Function<NDArray, NDArray> fn = NDArray::relu;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        assertEquals(1f, gradFn.apply(scalar(2f)).toFloatArray()[0], 1e-4);
        assertEquals(0f, gradFn.apply(scalar(-2f)).toFloatArray()[0], 1e-4);
    }

    @Test
    public void testSigmoidGrad() {
        // f(x) = sigmoid(x) ; f'(x) = sigmoid(x) * (1 - sigmoid(x))
        Function<NDArray, NDArray> fn = NDArray::sigmoid;
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray x = scalar(0.4f);
        float result = gradFn.apply(x).toFloatArray()[0];
        float s = (float) (1.0 / (1.0 + Math.exp(-0.4)));
        assertEquals(s * (1f - s), result, 1e-4);
    }

    @Test
    public void testDotGrad() {
        // f(A) = sum(A . B), A is (1,2), B is (2,1) fixed
        // d(sum(A.B))/dA = B^T broadcast across rows of A
        NDArray b = new ConcreteNDArray(new float[]{5f, 7f}, new Shape(2, 1));
        Function<NDArray, NDArray> fn = a -> a.dot(b).sum();
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);

        NDArray a = new ConcreteNDArray(new float[]{1f, 2f}, new Shape(1, 2));
        float[] result = gradFn.apply(a).toFloatArray();
        assertEquals(5f, result[0], 1e-4);
        assertEquals(7f, result[1], 1e-4);
    }
}
