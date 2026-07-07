package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.scalar;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@code JAX.vmap}, modeled on the batching cases in upstream
 * JAX's {@code tests/batching_test.py}: a function traced once against a
 * single example must produce the same per-example results when applied to
 * a whole batch at once.
 */
public class VmapTest {

    @Test
    public void testElementwiseVmapMatchesPerExampleLoop() {
        Function<NDArray, NDArray> fn = x -> x.mul(x).add(scalar(1));
        NDArray batch = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(4, 1));

        NDArray batched = JAX.vmap(fn).apply(batch);

        assertEquals(new Shape(4, 1), batched.shape());
        assertArrayEquals(new float[]{2, 5, 10, 17}, batched.toFloatArray(), 1e-5f);
    }

    @Test
    public void testVmapWithClosedOverConstant() {
        NDArray weights = new ConcreteNDArray(new float[]{2, 3, 4}, new Shape(3));
        Function<NDArray, NDArray> fn = x -> x.mul(weights).sum();

        // 2 examples, each of length 3
        NDArray batch = new ConcreteNDArray(new float[]{1, 1, 1, 2, 2, 2}, new Shape(2, 3));
        NDArray batched = JAX.vmap(fn).apply(batch);

        // example0: 1*2+1*3+1*4=9 ; example1: 2*2+2*3+2*4=18
        assertEquals(new Shape(2), batched.shape());
        assertArrayEquals(new float[]{9, 18}, batched.toFloatArray(), 1e-5f);
    }

    @Test
    public void testVmapMeanPerExample() {
        Function<NDArray, NDArray> fn = NDArray::mean;
        NDArray batch = new ConcreteNDArray(new float[]{1, 2, 3, 10, 20, 30}, new Shape(2, 3));

        NDArray batched = JAX.vmap(fn).apply(batch);

        assertArrayEquals(new float[]{2.0f, 20.0f}, batched.toFloatArray(), 1e-5f);
    }

    @Test
    public void testVmapDotWithSharedMatrix() {
        // Each example is a 1x2 row vector; shared (2x2) matrix is closed over (unbatched).
        NDArray sharedMatrix = new ConcreteNDArray(new float[]{1, 0, 0, 1}, new Shape(2, 2)); // identity
        Function<NDArray, NDArray> fn = x -> x.dot(sharedMatrix);

        NDArray batch = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(2, 1, 2));
        NDArray batched = JAX.vmap(fn).apply(batch);

        assertEquals(new Shape(2, 1, 2), batched.shape());
        assertArrayEquals(new float[]{1, 2, 3, 4}, batched.toFloatArray(), 1e-5f);
    }

    @Test
    public void testVmapMatchesManualPerExampleComputation() {
        Function<NDArray, NDArray> fn = x -> x.exp().sum();
        float[][] examples = {{0.1f, 0.2f}, {1.0f, -1.0f}, {0.0f, 3.0f}};
        float[] flat = new float[6];
        for (int i = 0; i < 3; i++) {
            flat[i * 2] = examples[i][0];
            flat[i * 2 + 1] = examples[i][1];
        }
        NDArray batch = new ConcreteNDArray(flat, new Shape(3, 2));

        NDArray batched = JAX.vmap(fn).apply(batch);

        float[] expected = new float[3];
        for (int i = 0; i < 3; i++) {
            expected[i] = (float) (Math.exp(examples[i][0]) + Math.exp(examples[i][1]));
        }
        assertArrayEquals(expected, batched.toFloatArray(), 1e-4f);
    }
}
