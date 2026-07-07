package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.Arrays;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Barrier-synchronized collective state shared across all shard threads of a
 * single pmap execution. Each operation uses a two-phase protocol: every shard
 * deposits its local data at its own slot, waits at a {@link CyclicBarrier}
 * until all shards have arrived, then reads the globally combined result.
 *
 * <p>A fresh {@code SharedCollective} is created per pmap call (forward and
 * backward each get their own instance). The {@link CyclicBarrier} resets
 * automatically after each await, so the same collective object can be used for
 * multiple sequential collective ops within the body function.
 */
final class SharedCollective {

    private final int numDevices;
    private final CyclicBarrier barrier;

    @SuppressWarnings("unchecked")
    private final AtomicReference<float[]>[] slots;

    @SuppressWarnings("unchecked")
    SharedCollective(int numDevices) {
        this.numDevices = numDevices;
        this.barrier = new CyclicBarrier(numDevices);
        this.slots = new AtomicReference[numDevices];
        for (int i = 0; i < numDevices; i++) slots[i] = new AtomicReference<>();
    }

    /**
     * All-reduce sum: shard {@code d} deposits {@code data}, all wait at the
     * barrier, then all read the element-wise sum across every shard's deposit.
     * Mirrors {@code jax.lax.psum} / NCCL {@code AllReduce(SUM)}.
     */
    float[] psum(int deviceIndex, float[] data) {
        slots[deviceIndex].set(data.clone());
        await();  // phase 1: all shards have deposited
        float[] result = new float[data.length];
        for (int d = 0; d < numDevices; d++) {
            float[] v = slots[d].get();
            for (int i = 0; i < result.length; i++) result[i] += v[i];
        }
        await();  // phase 2: all shards have finished reading; safe to reuse slots
        return result;
    }

    /**
     * All-gather: shard {@code d} deposits its slice, all wait, then all read
     * the full stacked result of shape {@code [numDevices, *shardShape]}.
     * Mirrors {@code jax.lax.all_gather} / NCCL {@code AllGather}.
     */
    NDArray allGather(int deviceIndex, NDArray shard) {
        Shape shardShape = shard.shape();
        slots[deviceIndex].set(shard.toFloatArray().clone());
        await();  // phase 1
        int stride = (int) shardShape.size();
        float[] out = new float[numDevices * stride];
        for (int d = 0; d < numDevices; d++) {
            System.arraycopy(slots[d].get(), 0, out, d * stride, stride);
        }
        int[] newDims = new int[shardShape.rank() + 1];
        newDims[0] = numDevices;
        System.arraycopy(shardShape.dimensions(), 0, newDims, 1, shardShape.rank());
        NDArray result = new ConcreteNDArray(out, new Shape(newDims));
        await();  // phase 2: all shards have finished reading
        return result;
    }

    private void await() {
        try {
            barrier.await();
        } catch (InterruptedException | BrokenBarrierException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Collective barrier interrupted or broken", e);
        }
    }
}
