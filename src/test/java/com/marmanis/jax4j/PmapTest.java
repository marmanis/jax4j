package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Lax;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@link JAX#pmap}, {@link Lax#psum}, and {@link Lax#allGather}:
 * axis splitting, collective ops, and autodiff integration.
 *
 * <p>All tests use two synthetic host devices ({@code new Device(null)}) so
 * they pass without a real GPU or TornadoVM runtime — the concurrent sharding
 * and collective barrier logic is exercised identically regardless.
 */
public class PmapTest {

    private static final int D = 2; // num devices for all tests
    private List<Device> devices;

    @BeforeEach
    void setUp() {
        // Two distinct host-backed devices (identity-equal, so they are "different").
        devices = List.of(new Device(null), new Device(null));
    }

    private static NDArray arr(float... vs) {
        return new ConcreteNDArray(vs, new Shape(vs.length));
    }

    // ---- basic forward ----

    @Test
    void pmapSquaresEachShardIndependently() {
        // Input [2, 3]: shard 0 = [1,2,3], shard 1 = [4,5,6]
        // pmap(x -> x*x): shard 0 → [1,4,9], shard 1 → [16,25,36]
        NDArray input = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray result = JAX.pmap(x -> x.mul(x), devices).apply(input);

        assertEquals(2, result.shape().rank());
        assertEquals(2, result.shape().dimensions()[0]);
        assertEquals(3, result.shape().dimensions()[1]);
        assertArrayEquals(new float[]{1, 4, 9, 16, 25, 36}, result.toFloatArray(), 1e-5f);
    }

    @Test
    void pmapScalarOutputStackedAcrossDevices() {
        // Each shard [3] sums to a scalar [1]; pmap stacks → [2, 1]
        NDArray input = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray result = JAX.pmap(NDArray::sum, devices).apply(input);

        assertEquals(2, result.shape().dimensions()[0]);
        assertArrayEquals(new float[]{6f, 15f}, result.toFloatArray(), 1e-5f);
    }

    // ---- psum collective ----

    @Test
    void psumReducesAcrossShards() {
        // Each shard has value [10, 20]; psum should give each shard [20, 40] (sum of both)
        NDArray input = new ConcreteNDArray(new float[]{10, 20, 10, 20}, new Shape(2, 2));
        NDArray result = JAX.pmap(x -> Lax.psum(x), devices).apply(input);

        // Both shards contribute [10,20], so each gets [20, 40]
        assertArrayEquals(new float[]{20, 40, 20, 40}, result.toFloatArray(), 1e-5f);
    }

    @Test
    void psumWithMeanGivesGlobalMean() {
        // Global mean over all elements using psum: local_sum / total_count
        // Shard 0: [1,2,3] → local_sum = 6; Shard 1: [4,5,6] → local_sum = 15
        // psum: both shards get 21; mean = 21/6 = 3.5
        NDArray input = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray total = new ConcreteNDArray(new float[]{6f}, new Shape(1));
        NDArray result = JAX.pmap(x -> Lax.psum(x.sum()).div(total), devices).apply(input);

        assertArrayEquals(new float[]{3.5f, 3.5f}, result.toFloatArray(), 1e-4f);
    }

    // ---- all_gather collective ----

    @Test
    void allGatherCollectsAllShardsOnEveryDevice() {
        // Shard 0: [1,2], Shard 1: [3,4]; all_gather → each device sees [[1,2],[3,4]]
        NDArray input = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(2, 2));
        NDArray result = JAX.pmap(x -> Lax.allGather(x, D).sum(), devices).apply(input);

        // Each shard sums all gathered [1,2,3,4] = 10
        assertArrayEquals(new float[]{10f, 10f}, result.toFloatArray(), 1e-5f);
    }

    // ---- autodiff through pmap ----

    @Test
    void gradOfPmapSquareIsLinear() {
        // f(x) = pmap(s -> s*s, devices)(x).sum() where x has shape [2, 3]
        // = sum(x^2); grad_x = 2*x, stacked as [2, 3]
        Function<NDArray, NDArray> fn = x -> JAX.pmap(s -> s.mul(s), devices).apply(x).sum();

        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        float fwd = fn.apply(x).toFloatArray()[0];
        assertEquals(1 + 4 + 9 + 16 + 25 + 36, fwd, 1e-4f);

        NDArray grad = JAX.grad(fn).apply(x);
        assertArrayEquals(new float[]{2, 4, 6, 8, 10, 12}, grad.toFloatArray(), 1e-4f);
    }

    @Test
    void gradOfPmapWithPsumAveragesGradients() {
        // f(x) = pmap(s -> psum(s*s) / scalar(2), devices)(x).sum()
        // Each shard i gets sum of all x_j^2 then divided by 2.
        // After pmap+sum: sum_i (psum(x_j^2) / 2) = 2 * (sum x_j^2) / 2 = sum x_j^2
        // grad_x_j = 2 * x_j (each x_j appears in both psums)
        NDArray two = new ConcreteNDArray(new float[]{2f}, new Shape(1));
        Function<NDArray, NDArray> fn = x ->
            JAX.pmap(s -> Lax.psum(s.mul(s).sum()).div(two), devices).apply(x).sum();

        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(2, 2));
        float fwd = fn.apply(x).toFloatArray()[0];
        // psum(s*s).sum() = (1+4+9+16) = 30, div 2 = 15; both shards get 15; sum = 30
        assertEquals(30f, fwd, 1e-3f);

        NDArray grad = JAX.grad(fn).apply(x);
        // Each x_i contributes to the psum seen by all D=2 shards, then div 2:
        // dL/dx_i = 2 (D shards each see x_i, psum forward doubles the gradient,
        // divided by 2 cancels that out). Net: grad = 2*x_i.
        assertArrayEquals(new float[]{2, 4, 6, 8}, grad.toFloatArray(), 1e-3f);
    }

    @Test
    void pmapComposesWithVmap() {
        // vmap over a batch of pmap inputs
        // Inner: pmap(x -> x*x, D) maps [D, k] → [D, k]
        // Outer: vmap over B batches of [D, k] → [B, D, k]
        NDArray batchInput = new ConcreteNDArray(
            new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new Shape(2, 2, 2));
        // each example has shape [2, 2]; pmap squares each shard [2] → [2]; output [2, 2]
        NDArray result = JAX.vmap(x -> JAX.pmap(s -> s.mul(s), devices).apply(x)).apply(batchInput);

        assertEquals(3, result.shape().rank());
        assertArrayEquals(new float[]{1, 4, 9, 16, 25, 36, 49, 64}, result.toFloatArray(), 1e-5f);
    }

    @Test
    void psumOutsidePmapIsIdentity() {
        // psum on a plain array (no PmapContext active) returns the array unchanged
        NDArray x = arr(3f, 7f);
        assertArrayEquals(x.toFloatArray(), Lax.psum(x).toFloatArray(), 1e-6f);
    }

    @Test
    void allGatherOutsidePmapPrependsOneDim() {
        NDArray x = arr(5f, 6f);
        NDArray result = Lax.allGather(x, 1);
        // shape [1, 2], data [5, 6]
        assertEquals(2, result.shape().rank());
        assertEquals(1, result.shape().dimensions()[0]);
        assertArrayEquals(new float[]{5f, 6f}, result.toFloatArray(), 1e-6f);
    }
}
