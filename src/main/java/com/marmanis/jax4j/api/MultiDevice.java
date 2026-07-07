package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

/**
 * Concurrent fan-out of independent work across multiple devices.
 *
 * <p>This is deliberately not {@code pmap}: there is no axis splitting, no
 * collective communication between devices, and no autodiff integration —
 * just the concurrent-dispatch primitive a future {@code pmap} would build on.
 * Each shard is placed on its device via {@link NDArray#to(Device)} before
 * {@code fn} runs, so device-checked ops inside {@code fn} (see
 * {@code ConcreteNDArray#elementwise}) dispatch to that device correctly.
 */
public final class MultiDevice {
    private MultiDevice() {}

    /**
     * Runs {@code fn} on each {@code (devices[i], shards[i])} pair concurrently
     * and returns the results in the same order as {@code shards}.
     */
    public static NDArray[] map(List<Device> devices, NDArray[] shards, Function<NDArray, NDArray> fn) {
        if (devices.size() != shards.length) {
            throw new IllegalArgumentException(
                "Number of devices (" + devices.size() + ") must match number of shards (" + shards.length + ")");
        }

        ExecutorService pool = Executors.newFixedThreadPool(devices.size());
        try {
            List<Future<NDArray>> futures = new ArrayList<>(shards.length);
            for (int i = 0; i < shards.length; i++) {
                NDArray placedShard = shards[i].to(devices.get(i));
                futures.add(pool.submit(() -> fn.apply(placedShard)));
            }

            NDArray[] results = new NDArray[shards.length];
            for (int i = 0; i < futures.size(); i++) {
                try {
                    results[i] = futures.get(i).get();
                } catch (ExecutionException e) {
                    throw new RuntimeException("MultiDevice.map task failed on device " + devices.get(i), e.getCause());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("MultiDevice.map interrupted", e);
                }
            }
            return results;
        } finally {
            pool.shutdown();
        }
    }
}
