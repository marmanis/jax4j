package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

public class GradTest {

    @Test
    public void testSimpleGrad() {
        // f(x) = x * x + x
        // f'(x) = 2x + 1
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(x);
        
        Function<NDArray, NDArray> gradFn = JAX.grad(fn);
        
        // x = 3.0
        NDArray arg = new ConcreteNDArray(new float[]{3.0f}, new Shape(1));
        NDArray grad = gradFn.apply(arg);
        
        // f'(3) = 2*3 + 1 = 7.0
        float[] result = grad.toFloatArray();
        assertEquals(1, result.length);
        assertEquals(7.0f, result[0], 1e-5);
    }

    @Test
    public void testSimpleJvp() {
        // f(x) = x * x + x
        // f'(x) = 2x + 1
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(x);

        // x = 3.0, tangent = 1.5
        NDArray arg = new ConcreteNDArray(new float[]{3.0f}, new Shape(1));
        NDArray tangent = new ConcreteNDArray(new float[]{1.5f}, new Shape(1));

        NDArray[] out = JAX.jvp(fn, arg, tangent);

        // primal = f(3) = 12.0
        assertArrayEquals(new float[]{12.0f}, out[0].toFloatArray(), 1e-5f);
        // tangent = f'(3) * 1.5 = (2 * 3 + 1) * 1.5 = 10.5
        assertArrayEquals(new float[]{10.5f}, out[1].toFloatArray(), 1e-5f);
    }

    @Test
    public void testSinJvp() {
        // f(x) = sin(x)
        // f'(x) = cos(x)
        Function<NDArray, NDArray> fn = NDArray::sin;

        // x = 0.0, tangent = 2.0
        NDArray arg = new ConcreteNDArray(new float[]{0.0f}, new Shape(1));
        NDArray tangent = new ConcreteNDArray(new float[]{2.0f}, new Shape(1));

        NDArray[] out = JAX.jvp(fn, arg, tangent);

        assertArrayEquals(new float[]{0.0f}, out[0].toFloatArray(), 1e-5f);
        assertArrayEquals(new float[]{2.0f}, out[1].toFloatArray(), 1e-5f);
    }

    @Test
    public void testDotJvp() {
        // Matrix multiplication JVP
        // f(X) = X . X
        // d(X.X) = X_t.X + X.X_t
        Function<NDArray, NDArray> fn = x -> x.dot(x);

        NDArray arg = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
        NDArray tangent = new ConcreteNDArray(new float[]{1f, 0f, 0f, 1f}, new Shape(2, 2)); // Identity matrix

        NDArray[] out = JAX.jvp(fn, arg, tangent);

        // primal = X.X = [7, 10, 15, 22]
        assertArrayEquals(new float[]{7f, 10f, 15f, 22f}, out[0].toFloatArray(), 1e-5f);
        // tangent = I.X + X.I = 2 * X = [2, 4, 6, 8]
        assertArrayEquals(new float[]{2f, 4f, 6f, 8f}, out[1].toFloatArray(), 1e-5f);
    }
}
