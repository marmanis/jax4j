package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Jaxpr;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

public class TracingTest {

    @Test
    public void testSimpleTrace() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(x);
        
        NDArray arg = new ConcreteNDArray(new float[]{1.0f}, new Shape(1));
        Jaxpr jaxpr = JAX.make_jaxpr(fn, arg);
        
        System.out.println(jaxpr);
        
        assertNotNull(jaxpr);
        assertEquals(1, jaxpr.inVars().size());
        assertEquals(1, jaxpr.outVars().size());
        assertEquals(2, jaxpr.equations().size());
        
        // Equation 1: v1 = mul v0, v0
        assertEquals("mul", jaxpr.equations().get(0).primitive().toString());
        // Equation 2: v2 = add v1, v0
        assertEquals("add", jaxpr.equations().get(1).primitive().toString());
    }
}
