package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Lax;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for {@link Lax}: {@code cond}/{@code whileLoop}/{@code scan}/{@code
 * foriLoop} both eagerly (outside any trace) and through {@code grad}, to
 * confirm they're real differentiable parts of the traced graph rather than
 * being unrolled away outside it.
 */
public class LaxControlFlowTest {

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }

    // ---- cond ----

    @Test
    public void condEagerSelectsBranchByPredicate() {
        Function<NDArray, NDArray> square = x -> x.mul(x);
        Function<NDArray, NDArray> negate = x -> x.mul(scalar(-1f));

        NDArray truthy = Lax.cond(scalar(1f), square, negate, scalar(3f));
        NDArray falsy = Lax.cond(scalar(0f), square, negate, scalar(3f));

        assertEquals(9f, truthy.toFloatArray()[0], 1e-5f);
        assertEquals(-3f, falsy.toFloatArray()[0], 1e-5f);
    }

    @Test
    public void condGradFollowsTakenBranchOnly() {
        // f(x) = pred ? x^2 : 3*x ; pred is fixed per call (not differentiated).
        Function<NDArray, NDArray> square = x -> x.mul(x);
        Function<NDArray, NDArray> timesThree = x -> x.mul(scalar(3f));

        Function<NDArray, NDArray> fTrue = x -> Lax.cond(scalar(1f), square, timesThree, x);
        Function<NDArray, NDArray> fFalse = x -> Lax.cond(scalar(0f), square, timesThree, x);

        float gradTrue = JAX.grad(fTrue).apply(scalar(4f)).toFloatArray()[0];
        float gradFalse = JAX.grad(fFalse).apply(scalar(4f)).toFloatArray()[0];

        assertEquals(8f, gradTrue, 1e-4f);  // d/dx x^2 at x=4
        assertEquals(3f, gradFalse, 1e-4f); // d/dx 3x
    }

    @Test
    public void condComposesWithSurroundingTrace() {
        // f(x) = (pred ? x^2 : x) + 1, pred depends on x being traced too.
        Function<NDArray, NDArray> square = y -> y.mul(y);
        Function<NDArray, NDArray> identity = y -> y;
        Function<NDArray, NDArray> fn = x -> Lax.cond(scalar(1f), square, identity, x).add(scalar(1f));

        float result = fn.apply(scalar(5f)).toFloatArray()[0];
        assertEquals(26f, result, 1e-4f);

        float grad = JAX.grad(fn).apply(scalar(5f)).toFloatArray()[0];
        assertEquals(10f, grad, 1e-4f); // d/dx (x^2 + 1) = 2x
    }

    // ---- while_loop ----

    @Test
    public void whileLoopEagerlyDoublesUntilThreshold() {
        // Double x until it's >= 100.
        Function<NDArray, NDArray> condFn = x -> scalar(x.toFloatArray()[0] < 100f ? 1f : 0f);
        Function<NDArray, NDArray> bodyFn = x -> x.mul(scalar(2f));

        NDArray result = Lax.whileLoop(condFn, bodyFn, scalar(3f));
        assertEquals(192f, result.toFloatArray()[0], 1e-4f); // 3 -> 6 -> 12 -> 24 -> 48 -> 96 -> 192
    }

    @Test
    public void whileLoopIsNotDifferentiable() {
        // condFn must itself be traceable (it's traced once via make_jaxpr, like
        // any cond/body function here): x * 0 is always falsy, regardless of x's
        // real value, so the loop body never runs and there's no risk of looping
        // forever once this is actually executed during the backward pass.
        Function<NDArray, NDArray> condFn = x -> x.mul(scalar(0f));
        Function<NDArray, NDArray> bodyFn = x -> x.add(scalar(1f));
        Function<NDArray, NDArray> fn = x -> Lax.whileLoop(condFn, bodyFn, x);

        assertThrows(UnsupportedOperationException.class, () -> JAX.grad(fn).apply(scalar(5f)));
    }

    // ---- fori_loop ----

    @Test
    public void foriLoopEagerlyAccumulates() {
        NDArray result = Lax.foriLoop(0, 5, (i, acc) -> acc.add(scalar(i)), scalar(0f));
        assertEquals(0 + 1 + 2 + 3 + 4, result.toFloatArray()[0], 1e-4f);
    }

    @Test
    public void foriLoopIsDifferentiable() {
        // f(x) = x repeatedly multiplied by 2, five times -> 32x ; f'(x) = 32
        Function<NDArray, NDArray> fn = x -> Lax.foriLoop(0, 5, (i, acc) -> acc.mul(scalar(2f)), x);

        assertEquals(32f * 3f, fn.apply(scalar(3f)).toFloatArray()[0], 1e-4f);
        assertEquals(32f, JAX.grad(fn).apply(scalar(3f)).toFloatArray()[0], 1e-4f);
    }

    // ---- scan ----

    @Test
    public void scanEagerlyComputesRunningSumAndPerStepSquares() {
        // carry = running sum, y = carry^2 at each step
        java.util.function.BiFunction<NDArray, NDArray, NDArray[]> stepFn = (carry, x) -> {
            NDArray newCarry = carry.add(x);
            return new NDArray[]{newCarry, newCarry.mul(newCarry)};
        };
        NDArray xs = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(4, 1));

        Lax.ScanResult result = Lax.scan(stepFn, scalar(0f), xs);

        assertEquals(10f, result.carry().toFloatArray()[0], 1e-4f); // 1+2+3+4
        assertArrayEquals(new float[]{1, 9, 36, 100}, result.ys().toFloatArray(), 1e-4f); // running sums squared: 1,3,6,10
    }

    @Test
    public void scanGradMatchesManualUnrolledComputation() {
        // f(x0) = scan(carry -> carry*x_t, x0, [2,3,4]).carry == x0 * 2 * 3 * 4
        // d/dx0 f = 2*3*4 = 24
        java.util.function.BiFunction<NDArray, NDArray, NDArray[]> stepFn =
            (carry, x) -> new NDArray[]{carry.mul(x), carry};
        NDArray xs = new ConcreteNDArray(new float[]{2, 3, 4}, new Shape(3, 1));

        Function<NDArray, NDArray> fn = x0 -> Lax.scan(stepFn, x0, xs).carry();

        float forward = fn.apply(scalar(5f)).toFloatArray()[0];
        assertEquals(5f * 2f * 3f * 4f, forward, 1e-4f);

        float grad = JAX.grad(fn).apply(scalar(5f)).toFloatArray()[0];
        assertEquals(2f * 3f * 4f, grad, 1e-4f);
    }

    @Test
    public void scanGradThroughStackedYsSumsContributionsAcrossSteps() {
        // f(x0) = sum_t (x0 + t)^2 for t in {1,2,3}, carry unused beyond threading.
        // d/dx0 f = sum_t 2*(x0+t)
        java.util.function.BiFunction<NDArray, NDArray, NDArray[]> stepFn = (carry, t) -> {
            NDArray val = carry.add(t);
            return new NDArray[]{carry, val.mul(val)};
        };
        NDArray ts = new ConcreteNDArray(new float[]{1, 2, 3}, new Shape(3, 1));

        Function<NDArray, NDArray> fn = x0 -> Lax.scan(stepFn, x0, ts).ys().sum();

        float x0 = 5f;
        float expectedForward = (float) ((x0 + 1) * (x0 + 1) + (x0 + 2) * (x0 + 2) + (x0 + 3) * (x0 + 3));
        assertEquals(expectedForward, fn.apply(scalar(x0)).toFloatArray()[0], 1e-3f);

        float expectedGrad = 2 * (x0 + 1) + 2 * (x0 + 2) + 2 * (x0 + 3);
        assertEquals(expectedGrad, JAX.grad(fn).apply(scalar(x0)).toFloatArray()[0], 1e-3f);
    }
}
