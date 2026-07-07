package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link Random} (explicit-key sampling) and {@link Nn#dropout}.
 * Statistical assertions use a large sample count and generous tolerances to
 * avoid flakiness while still catching gross errors (wrong distribution,
 * wrong scale, non-determinism).
 */
public class RandomTest {

    private static final int N = 20_000;

    // ---- determinism / splitting ----

    @Test
    void sameKeyGivesIdenticalOutput() {
        PRNGKey key = PRNGKey.key(42);
        NDArray a = Random.uniform(key, new Shape(10));
        NDArray b = Random.uniform(key, new Shape(10));
        assertArrayEquals(a.toFloatArray(), b.toFloatArray(), 0f);
    }

    @Test
    void differentKeysGiveDifferentOutput() {
        NDArray a = Random.uniform(PRNGKey.key(1), new Shape(10));
        NDArray b = Random.uniform(PRNGKey.key(2), new Shape(10));
        assertFalse(java.util.Arrays.equals(a.toFloatArray(), b.toFloatArray()));
    }

    @Test
    void splitIsDeterministicAndChildrenDiffer() {
        PRNGKey key = PRNGKey.key(7);
        PRNGKey[] first = Random.split(key);
        PRNGKey[] second = Random.split(key);
        assertEquals(first[0].state(), second[0].state());
        assertEquals(first[1].state(), second[1].state());
        assertFalse(first[0].state() == first[1].state());
    }

    @Test
    void splitNProducesNDistinctKeys() {
        PRNGKey[] keys = Random.split(PRNGKey.key(99), 5);
        assertEquals(5, keys.length);
        for (int i = 0; i < keys.length; i++) {
            for (int j = i + 1; j < keys.length; j++) {
                assertFalse(keys[i].state() == keys[j].state());
            }
        }
    }

    // ---- distributions ----

    @Test
    void uniformIsWithinBoundsAndApproximatelyCentered() {
        NDArray x = Random.uniform(PRNGKey.key(1), new Shape(N), -2f, 6f);
        float sum = 0f;
        for (float v : x.toFloatArray()) {
            assertTrue(v >= -2f && v < 6f);
            sum += v;
        }
        float mean = sum / N;
        assertEquals(2f, mean, 0.15f); // (-2+6)/2 = 2
    }

    @Test
    void normalHasApproximatelyZeroMeanUnitVariance() {
        NDArray x = Random.normal(PRNGKey.key(2), new Shape(N));
        float[] data = x.toFloatArray();
        double sum = 0, sumSq = 0;
        for (float v : data) { sum += v; sumSq += (double) v * v; }
        double mean = sum / N;
        double variance = sumSq / N - mean * mean;
        assertEquals(0.0, mean, 0.05);
        assertEquals(1.0, variance, 0.15);
    }

    @Test
    void bernoulliMatchesRequestedProbability() {
        NDArray x = Random.bernoulli(PRNGKey.key(3), 0.3f, new Shape(N));
        assertEquals(com.marmanis.jax4j.core.DType.BOOL, x.dtype());
        float ones = 0;
        for (boolean v : x.toBoolArray()) {
            if (v) ones++;
        }
        assertEquals(0.3f, ones / N, 0.02f);
    }

    @Test
    void permutationIsAValidPermutation() {
        int n = 200;
        NDArray perm = Random.permutation(PRNGKey.key(4), n);
        assertEquals(com.marmanis.jax4j.core.DType.INT32, perm.dtype());
        boolean[] seen = new boolean[n];
        for (int idx : perm.toIntArray()) {
            assertTrue(idx >= 0 && idx < n);
            assertFalse(seen[idx], "index " + idx + " appeared twice");
            seen[idx] = true;
        }
        for (boolean s : seen) assertTrue(s);
    }

    // ---- initializers ----

    @Test
    void glorotUniformRespectsLimit() {
        int fanIn = 100, fanOut = 50;
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        NDArray w = Random.glorotUniform(PRNGKey.key(5), new Shape(fanIn, fanOut), fanIn, fanOut);
        for (float v : w.toFloatArray()) {
            assertTrue(v >= -limit && v < limit);
        }
    }

    @Test
    void heNormalScalesByExpectedStd() {
        int fanIn = 256;
        float expectedStd = (float) Math.sqrt(2.0 / fanIn);
        NDArray w = Random.heNormal(PRNGKey.key(6), new Shape(N), fanIn);
        double sumSq = 0;
        for (float v : w.toFloatArray()) sumSq += (double) v * v;
        double std = Math.sqrt(sumSq / N);
        assertEquals(expectedStd, std, expectedStd * 0.15);
    }

    // ---- dropout ----

    @Test
    void dropoutEvalModeIsPassthrough() {
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(4));
        NDArray out = Nn.dropout(x, PRNGKey.key(1), 0.5f, false);
        assertArrayEquals(new float[]{1, 2, 3, 4}, out.toFloatArray(), 1e-6f);
    }

    @Test
    void dropoutTrainModeZeroesApproximatelyRateFraction() {
        NDArray x = new ConcreteNDArray(onesArray(N), new Shape(N));
        NDArray out = Nn.dropout(x, PRNGKey.key(8), 0.4f, true);
        int zeros = 0;
        float keepScale = 1f / 0.6f;
        for (float v : out.toFloatArray()) {
            if (v == 0f) zeros++;
            else assertEquals(keepScale, v, 1e-4f);
        }
        assertEquals(0.4f, zeros / (float) N, 0.02f);
    }

    @Test
    void dropoutGradientIsZeroAtDroppedAndScaledAtKept() {
        NDArray x = new ConcreteNDArray(new float[]{1, 1, 1, 1, 1, 1, 1, 1}, new Shape(8));
        PRNGKey key = PRNGKey.key(123);
        float rate = 0.5f;

        // Determine which positions get dropped for this key/shape/rate up front.
        NDArray mask = Random.bernoulli(key, 1f - rate, x.shape());
        boolean[] m = mask.toBoolArray();

        Function<NDArray, NDArray> fn = v -> Nn.dropout(v, key, rate, true).sum();
        NDArray grad = JAX.grad(fn).apply(x);
        float[] g = grad.toFloatArray();

        float expectedScale = 1f / (1f - rate);
        for (int i = 0; i < g.length; i++) {
            if (!m[i]) assertEquals(0f, g[i], 1e-6f);
            else assertEquals(expectedScale, g[i], 1e-4f);
        }
    }

    private static float[] onesArray(int n) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, 1f);
        return a;
    }
}
