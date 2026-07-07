package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Jaxpr;
import org.junit.jupiter.api.Test;

import java.util.function.BiFunction;
import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.vector;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Structural and transformation-composition tests for the {@code JAX} entry
 * point, modeled on the role of upstream JAX's {@code tests/api_test.py}:
 * checking the shape of traced Jaxprs and the composability of transformations
 * rather than individual primitive math (covered by {@link LaxAutodiffTest}).
 */
public class ApiTest {

    @Test
    public void testMakeJaxprStructure() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(x);
        Jaxpr jaxpr = JAX.make_jaxpr(fn, vector(1.0f));

        assertEquals(1, jaxpr.inVars().size());
        assertEquals(1, jaxpr.outVars().size());
        assertEquals(2, jaxpr.equations().size());
        assertEquals("mul", jaxpr.equations().get(0).primitive().toString());
        assertEquals("add", jaxpr.equations().get(1).primitive().toString());
    }

    @Test
    public void testMakeJaxprTwoArgs() {
        BiFunction<NDArray, NDArray, NDArray> fn = (a, b) -> a.add(b).mul(a);
        Jaxpr jaxpr = JAX.make_jaxpr(fn, vector(1.0f), vector(2.0f));

        assertEquals(2, jaxpr.inVars().size());
        assertEquals(2, jaxpr.equations().size());
    }

    @Test
    public void testJaxprRecordsConstantsUsedInTrace() {
        // The literal vector(1, 2, 3) is not one of the traced inputs, so it
        // must show up as a const in the jaxpr rather than vanish.
        Function<NDArray, NDArray> fn = x -> x.add(vector(1, 2, 3));
        Jaxpr jaxpr = JAX.make_jaxpr(fn, vector(0, 0, 0));

        assertEquals(1, jaxpr.consts().size());
    }

    @Test
    public void testJitMatchesEagerExecution() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(x.sin());
        NDArray x = vector(0.3f, 1.7f, -2.1f);

        NDArray eager = fn.apply(x);
        NDArray jitted = JAX.jit(fn).apply(x);

        assertArrayEquals(eager.toFloatArray(), jitted.toFloatArray(), 1e-5f);
    }

    @Test
    public void testGradIsComposableWithOrdinaryFunctions() {
        // grad should compose transparently with surrounding host-side code,
        // not just appear at the top level of an expression.
        Function<NDArray, NDArray> square = x -> x.mul(x);
        Function<NDArray, NDArray> gradSquare = JAX.grad(square);

        NDArray x = vector(4.0f);
        NDArray doubled = gradSquare.apply(x).add(gradSquare.apply(x));

        assertEquals(16.0f, doubled.toFloatArray()[0], 1e-5f); // 2 * (2*4)
    }

    @Test
    public void testGradPreservesInputShape() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).sum();
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));

        NDArray grad = JAX.grad(fn).apply(x);

        assertEquals(x.shape(), grad.shape());
    }

    @Test
    public void testTracingDoesNotMutateConcreteInputs() {
        NDArray x = vector(1.0f, 2.0f, 3.0f);
        float[] before = x.toFloatArray().clone();

        JAX.make_jaxpr(v -> v.mul(v), x);

        assertArrayEquals(before, x.toFloatArray());
    }
}
