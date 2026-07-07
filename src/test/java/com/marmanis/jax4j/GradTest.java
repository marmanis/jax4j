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
    public void testComposeGrad() {
        // f(x) = x * x
        // g(x) = f(f(x)) = x^4
        // g'(x) = 4x^3
        Function<NDArray, NDArray> f = x -> x.mul(x);
        Function<NDArray, NDArray> g = x -> f.apply(f.apply(x));
        
        Function<NDArray, NDArray> gradG = JAX.grad(g);
        
        // x = 2.0
        NDArray arg = new ConcreteNDArray(new float[]{2.0f}, new Shape(1));
        NDArray grad = gradG.apply(arg);
        
        // g'(2) = 4 * 2^3 = 4 * 8 = 32.0
        float[] result = grad.toFloatArray();
        assertEquals(32.0f, result[0], 1e-5);
    }
}
