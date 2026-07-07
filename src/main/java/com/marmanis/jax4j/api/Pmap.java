package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Jaxpr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

/**
 * Parallel-map execution kernel. All methods are package-private; the public
 * entry point is {@link JAX#pmap}.
 *
 * <p>Three execution modes share the same split/stack helpers:
 * <ol>
 *   <li>{@link #runForward} — eager forward pass (fn applied to concrete shards)</li>
 *   <li>{@link #runForwardJaxpr} — re-interpreted forward during {@code grad}'s
 *       backward (needed to reconstitute forward values for PSUM/ALL_GATHER)</li>
 *   <li>{@link #runBackwardJaxpr} — one backward-interpret per shard, concurrent
 *       so that collective ops inside the body's backward pass can synchronize</li>
 * </ol>
 *
 * <p>Every shard runs in its own thread with a {@link PmapContext} set, giving
 * collective ops access to the {@link SharedCollective} barrier.
 */
final class Pmap {
    private Pmap() {}

    /**
     * Splits a {@code [D, *rest]} input into {@code D} shards of shape
     * {@code *rest}. The leading dimension must equal {@code numDevices}.
     */
    static NDArray[] split(NDArray input, int numDevices) {
        int[] dims = input.shape().dimensions();
        if (dims.length == 0 || dims[0] != numDevices) {
            throw new IllegalArgumentException(
                "pmap: input leading dim " + (dims.length > 0 ? dims[0] : 0)
                    + " must equal numDevices " + numDevices);
        }
        Shape shardShape = new Shape(Arrays.copyOfRange(dims, 1, dims.length));
        float[] data = input.toFloatArray();
        int stride = (int) shardShape.size();
        NDArray[] shards = new NDArray[numDevices];
        for (int i = 0; i < numDevices; i++) {
            shards[i] = new ConcreteNDArray(Arrays.copyOfRange(data, i * stride, (i + 1) * stride), shardShape);
        }
        return shards;
    }

    /**
     * Stacks {@code D} equal-shaped shards into {@code [D, *shardShape]}.
     */
    static NDArray stack(NDArray[] shards) {
        Shape shardShape = shards[0].shape();
        int[] outDims = new int[shardShape.rank() + 1];
        outDims[0] = shards.length;
        System.arraycopy(shardShape.dimensions(), 0, outDims, 1, shardShape.rank());
        float[] out = new float[shards.length * (int) shardShape.size()];
        int stride = (int) shardShape.size();
        for (int i = 0; i < shards.length; i++) {
            System.arraycopy(shards[i].toFloatArray(), 0, out, i * stride, stride);
        }
        return new ConcreteNDArray(out, new Shape(outDims));
    }

    /** Eager forward: split input, apply fn to each shard in parallel, stack results. */
    static NDArray runForward(Function<NDArray, NDArray> fn, NDArray input, List<Device> devices) {
        int D = devices.size();
        NDArray[] inputShards = split(input, D);
        SharedCollective collective = new SharedCollective(D);
        NDArray[] results = new NDArray[D];
        runParallel(D, i -> {
            PmapContext.set(new PmapContext(i, D, collective));
            try {
                results[i] = fn.apply(inputShards[i].to(devices.get(i)));
            } finally {
                PmapContext.clear();
            }
        });
        return stack(results);
    }

    /**
     * Runs a pre-traced body {@link Jaxpr} on each shard in parallel and stacks
     * the results. Used by {@link Grad} when re-interpreting a PMAP equation
     * during the backward pass.
     */
    static NDArray runForwardJaxpr(Jaxpr bodyJaxpr, List<Device> devices, NDArray[] inputShards) {
        int D = devices.size();
        SharedCollective collective = new SharedCollective(D);
        NDArray[] results = new NDArray[D];
        runParallel(D, i -> {
            PmapContext.set(new PmapContext(i, D, collective));
            try {
                results[i] = Grad.forwardInterpret(bodyJaxpr, List.of(inputShards[i])).get(0);
            } finally {
                PmapContext.clear();
            }
        });
        return stack(results);
    }

    /**
     * Runs the backward pass through a body {@link Jaxpr} for each shard in
     * parallel, returning one input-gradient shard per device. Concurrent
     * execution is required so that collective ops (PSUM, ALL_GATHER) inside
     * the body's backward re-interpretation can synchronize across threads.
     */
    static NDArray[] runBackwardJaxpr(Jaxpr bodyJaxpr, List<Device> devices,
                                       NDArray[] inputShards, NDArray[] gOutShards) {
        int D = devices.size();
        SharedCollective collective = new SharedCollective(D);
        NDArray[] gInputShards = new NDArray[D];
        runParallel(D, i -> {
            PmapContext.set(new PmapContext(i, D, collective));
            try {
                gInputShards[i] = Grad.backwardInterpret(
                    bodyJaxpr, List.of(inputShards[i]), List.of(gOutShards[i])).get(0);
            } finally {
                PmapContext.clear();
            }
        });
        return gInputShards;
    }

    private static void runParallel(int count, IntConsumerThrows task) {
        List<Callable<Void>> callables = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            final int idx = i;
            callables.add(() -> {
                try {
                    task.accept(idx);
                } catch (Exception e) {
                    throw e;
                } catch (Throwable t) {
                    throw new RuntimeException(t);
                }
                return null;
            });
        }
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<Void>> futures = pool.invokeAll(callables);
            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof RuntimeException re) throw re;
                    if (cause instanceof Error err) throw err;
                    throw new RuntimeException("pmap shard execution failed", cause);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("pmap parallel execution interrupted", e);
        }
    }

    @FunctionalInterface
    interface IntConsumerThrows {
        void accept(int i) throws Throwable;
    }
}
