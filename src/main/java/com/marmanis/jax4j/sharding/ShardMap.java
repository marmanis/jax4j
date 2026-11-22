package com.marmanis.jax4j.sharding;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

/**
 * SPMD execution transformation mapping functions across a {@link DeviceMesh} according
 * to {@link PartitionSpec}s, mirroring {@code jax.experimental.shard_map}.
 *
 * <p>Decomposes global arrays into local device shards along partitioned axes, invokes
 * the function on each accelerator/device concurrently, and reconstitutes the output
 * according to the output partition specification.
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class ShardMap {

    private ShardMap() {}

    /**
     * Executes {@code fn} in SPMD fashion across {@code mesh}.
     */
    public static Function<NDArray, NDArray> shardMap(
            Function<NDArray, NDArray> fn,
            DeviceMesh mesh,
            PartitionSpec inSpec,
            PartitionSpec outSpec) {

        return (input) -> {
            Device[] devices = mesh.devices();
            int numDevices = devices.length;

            // 1. Shard input array across devices according to inSpec
            NDArray[] localShards = shardArray(input, mesh, inSpec);

            // 2. Dispatch concurrently to all devices in the mesh
            ExecutorService pool = Executors.newFixedThreadPool(numDevices);
            NDArray[] outShards = new NDArray[numDevices];
            try {
                List<Future<NDArray>> futures = new ArrayList<>(numDevices);
                for (int i = 0; i < numDevices; i++) {
                    final int devIdx = i;
                    NDArray placedShard = localShards[devIdx].to(devices[devIdx]);
                    futures.add(pool.submit(() -> fn.apply(placedShard)));
                }

                for (int i = 0; i < numDevices; i++) {
                    outShards[i] = futures.get(i).get();
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("shard_map execution failed across mesh", e);
            } finally {
                pool.shutdown();
            }

            // 3. Reconstitute the full array from outShards according to outSpec
            return gatherArray(outShards, mesh, outSpec);
        };
    }

    /**
     * Shards a global NDArray into local per-device arrays based on the mesh partition spec.
     */
    public static NDArray[] shardArray(NDArray input, DeviceMesh mesh, PartitionSpec spec) {
        Device[] devices = mesh.devices();
        int numDevices = devices.length;

        // Check if any axis is partitioned
        int shardedDim = -1;
        String meshAxis = null;
        for (int d = 0; d < spec.rank(); d++) {
            if (spec.get(d) != null) {
                shardedDim = d;
                meshAxis = spec.get(d);
                break;
            }
        }

        if (shardedDim == -1) {
            // Replicated across all devices
            NDArray[] shards = new NDArray[numDevices];
            for (int i = 0; i < numDevices; i++) shards[i] = input;
            return shards;
        }

        int numShards = mesh.axisSize(meshAxis);
        int[] inDims = input.shape().dimensions();
        if (inDims[shardedDim] % numShards != 0) {
            throw new IllegalArgumentException("Dimension " + shardedDim + " size " + inDims[shardedDim]
                    + " is not divisible by mesh axis " + meshAxis + " size " + numShards);
        }

        int shardDimSize = inDims[shardedDim] / numShards;
        int rank = inDims.length;
        NDArray[] shards = new NDArray[numDevices];

        // Slice along the sharded dimension
        int[] starts = new int[rank];
        int[] stops = new int[rank];
        int[] steps = new int[rank];
        Arrays.fill(steps, 1);
        for (int i = 0; i < rank; i++) stops[i] = inDims[i];

        for (int s = 0; s < numShards; s++) {
            starts[shardedDim] = s * shardDimSize;
            stops[shardedDim] = (s + 1) * shardDimSize;
            NDArray shard = input.slice(starts, stops, steps);
            // Assign to matching devices along this mesh axis
            for (int dev = s; dev < numDevices; dev += numShards) {
                shards[dev] = shard;
            }
        }

        return shards;
    }

    /**
     * Reconstitutes local per-device arrays into a single global array.
     */
    public static NDArray gatherArray(NDArray[] shards, DeviceMesh mesh, PartitionSpec spec) {
        int shardedDim = -1;
        String meshAxis = null;
        for (int d = 0; d < spec.rank(); d++) {
            if (spec.get(d) != null) {
                shardedDim = d;
                meshAxis = spec.get(d);
                break;
            }
        }

        if (shardedDim == -1) {
            // Replicated: all shards are identical, return the first
            return shards[0];
        }

        int numShards = mesh.axisSize(meshAxis);
        List<NDArray> uniqueShards = new ArrayList<>(numShards);
        for (int i = 0; i < numShards; i++) {
            uniqueShards.add(shards[i]);
        }

        return com.marmanis.jax4j.api.Numpy.concatenate(uniqueShards, shardedDim);
    }
}
