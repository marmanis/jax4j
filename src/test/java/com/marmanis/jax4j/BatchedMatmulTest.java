package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BatchedMatmulTest {

    private static float[] randFloats(int n, long seed) {
        java.util.Random r = new java.util.Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = r.nextFloat() * 2f - 1f;
        return out;
    }

    private static NDArray manualBatchMatmul(NDArray a, NDArray b) {
        int[] ad = a.shape().dimensions();
        int[] bd = b.shape().dimensions();
        int B = ad[0], M = ad[1], K = ad[2], N = bd[2];
        float[] av = a.toFloatArray();
        float[] bv = b.toFloatArray();
        float[] out = new float[B * M * N];
        for (int batch = 0; batch < B; batch++) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float s = 0;
                    for (int k = 0; k < K; k++) s += av[batch * M * K + i * K + k] * bv[batch * K * N + k * N + j];
                    out[batch * M * N + i * N + j] = s;
                }
        }
        return new ConcreteNDArray(out, new Shape(B, M, N));
    }

    @Test
    public void rank3ByRank3() {
        NDArray a = new ConcreteNDArray(randFloats(2 * 3 * 4, 1), new Shape(2, 3, 4));
        NDArray b = new ConcreteNDArray(randFloats(2 * 4 * 5, 2), new Shape(2, 4, 5));
        NDArray r = a.matmul(b);
        assertEquals(new Shape(2, 3, 5), r.shape());
        assertArrayEquals(manualBatchMatmul(a, b).toFloatArray(), r.toFloatArray(), 1e-4f);
    }

    @Test
    public void rank4ByRank4() {
        NDArray q = new ConcreteNDArray(randFloats(2 * 3 * 4 * 5, 10), new Shape(2, 3, 4, 5));
        NDArray k = new ConcreteNDArray(randFloats(2 * 3 * 5 * 6, 11), new Shape(2, 3, 5, 6));
        NDArray r = q.matmul(k);
        assertEquals(new Shape(2, 3, 4, 6), r.shape());
    }

    @Test
    public void broadcastNoBatchOnRhs() {
        // [B, M, K] @ [K, N] should broadcast RHS across B batch positions.
        NDArray a = new ConcreteNDArray(randFloats(3 * 2 * 4, 20), new Shape(3, 2, 4));
        NDArray b = new ConcreteNDArray(randFloats(4 * 5, 21), new Shape(4, 5));
        NDArray r = a.matmul(b);
        assertEquals(new Shape(3, 2, 5), r.shape());
    }

    @Test
    public void gradientRank3() {
        // f(a, b) = sum(matmul(a, b))
        BiFunction<NDArray, NDArray, NDArray> fn = (a, b) -> a.matmul(b).sum();
        NDArray a = new ConcreteNDArray(randFloats(2 * 3 * 4, 30), new Shape(2, 3, 4));
        NDArray b = new ConcreteNDArray(randFloats(2 * 4 * 3, 31), new Shape(2, 4, 3));
        NDArray[] grads = JAX.gradBoth(fn).apply(a, b);
        // Numerical check on both operands.
        float h = 1e-2f;
        float[] av = a.toFloatArray();
        for (int i = 0; i < av.length; i++) {
            float[] ap = av.clone(); ap[i] += h;
            float[] am = av.clone(); am[i] -= h;
            float fp = fn.apply(new ConcreteNDArray(ap, a.shape()), b).toFloatArray()[0];
            float fm = fn.apply(new ConcreteNDArray(am, a.shape()), b).toFloatArray()[0];
            assertEquals((fp - fm) / (2 * h), grads[0].toFloatArray()[i], 1e-2f);
        }
        float[] bv = b.toFloatArray();
        for (int i = 0; i < bv.length; i++) {
            float[] bp = bv.clone(); bp[i] += h;
            float[] bm = bv.clone(); bm[i] -= h;
            float fp = fn.apply(a, new ConcreteNDArray(bp, b.shape())).toFloatArray()[0];
            float fm = fn.apply(a, new ConcreteNDArray(bm, b.shape())).toFloatArray()[0];
            assertEquals((fp - fm) / (2 * h), grads[1].toFloatArray()[i], 1e-2f);
        }
    }

    @Test
    public void gradientBroadcastBatch() {
        // [B, M, K] @ [K, N] : gradient must sum-reduce over B for the RHS.
        BiFunction<NDArray, NDArray, NDArray> fn = (a, b) -> a.matmul(b).sum();
        NDArray a = new ConcreteNDArray(randFloats(3 * 2 * 4, 40), new Shape(3, 2, 4));
        NDArray b = new ConcreteNDArray(randFloats(4 * 5, 41), new Shape(4, 5));
        NDArray[] grads = JAX.gradBoth(fn).apply(a, b);
        assertEquals(a.shape(), grads[0].shape());
        assertEquals(b.shape(), grads[1].shape());
        // Numerical check just on b.
        float h = 1e-2f;
        float[] bv = b.toFloatArray();
        for (int i = 0; i < bv.length; i++) {
            float[] bp = bv.clone(); bp[i] += h;
            float[] bm = bv.clone(); bm[i] -= h;
            float fp = fn.apply(a, new ConcreteNDArray(bp, b.shape())).toFloatArray()[0];
            float fm = fn.apply(a, new ConcreteNDArray(bm, b.shape())).toFloatArray()[0];
            assertEquals((fp - fm) / (2 * h), grads[1].toFloatArray()[i], 1e-2f);
        }
    }
}
