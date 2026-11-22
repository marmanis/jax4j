package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.sharding.DeviceMesh;
import com.marmanis.jax4j.sharding.NamedSharding;
import com.marmanis.jax4j.sharding.PartitionSpec;
import com.marmanis.jax4j.sharding.ShardMap;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Tests for Phase 3: Next-Gen Distributed Compute & Named Sharding (SPMD).
 */
public class Phase3FeaturesTest {

    @Test
    public void testDeviceMeshCoordinates() {
        Device d0 = Device.host();
        Device d1 = Device.host();
        Device d2 = Device.host();
        Device d3 = Device.host();

        DeviceMesh mesh = new DeviceMesh(
                new Device[]{d0, d1, d2, d3},
                new int[]{2, 2},
                new String[]{"data", "model"}
        );

        assertEquals(4, mesh.numDevices());
        assertEquals(2, mesh.axisSize("data"));
        assertEquals(2, mesh.axisSize("model"));

        assertEquals(d0, mesh.deviceAt(0, 0));
        assertEquals(d1, mesh.deviceAt(0, 1));
        assertEquals(d2, mesh.deviceAt(1, 0));
        assertEquals(d3, mesh.deviceAt(1, 1));
    }

    @Test
    public void testNamedShardingCreation() {
        DeviceMesh mesh = new DeviceMesh(new Device[]{Device.host(), Device.host()}, "data");
        PartitionSpec spec = PartitionSpec.of("data", null);
        NamedSharding sharding = NamedSharding.of(mesh, spec);

        assertNotNull(sharding.mesh());
        assertEquals(spec, sharding.spec());
        assertEquals("data", sharding.spec().get(0));
        assertEquals(null, sharding.spec().get(1));
    }

    @Test
    public void testShardMapSPMDExecution() {
        DeviceMesh mesh = new DeviceMesh(new Device[]{Device.host(), Device.host()}, "data");
        PartitionSpec inSpec = PartitionSpec.of("data");
        PartitionSpec outSpec = PartitionSpec.of("data");

        // Global input: [4 elements] -> sharded into 2 shards of [2 elements] each.
        // Place on host so the shards and the constant below share a device; under
        // TornadoVM, Device.defaultDevice() is a GPU, and mixing it with the
        // host-only mesh raises IllegalStateException in elementwise ops.
        NDArray input = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(4), DType.FLOAT32, Device.host());

        // SPMD function: computes x * 10 elementwise on each shard. Scalar lives
        // on the shard's device (host), matching x after ShardMap places it there.
        Function<NDArray, NDArray> fn = x ->
                x.mul(new ConcreteNDArray(new float[]{10f}, new Shape(1), DType.FLOAT32, Device.host()));

        Function<NDArray, NDArray> shardedFn = JAX.shard_map(fn, mesh, inSpec, outSpec);
        NDArray output = shardedFn.apply(input);

        assertEquals(new Shape(4), output.shape());
        assertArrayEquals(new float[]{10f, 20f, 30f, 40f}, output.toFloatArray());
    }

    @Test
    public void testShardMap2DMatrixRows() {
        DeviceMesh mesh = new DeviceMesh(new Device[]{Device.host(), Device.host()}, "data");
        PartitionSpec inSpec = PartitionSpec.of("data", null);
        PartitionSpec outSpec = PartitionSpec.of("data", null);

        // Global input: [4, 2] matrix sharded along rows across 2 devices -> [2, 2] per device
        float[] data = new float[]{
                1f, 1f,
                2f, 2f,
                3f, 3f,
                4f, 4f
        };
        NDArray input = new ConcreteNDArray(data, new Shape(4, 2), DType.FLOAT32, Device.host());

        // Function: add 100 to all local elements. Constant placed on host to
        // match the host-only mesh (see testShardMapSPMDExecution for rationale).
        Function<NDArray, NDArray> fn = x ->
                x.add(new ConcreteNDArray(new float[]{100f}, new Shape(1), DType.FLOAT32, Device.host()));

        NDArray output = ShardMap.shardMap(fn, mesh, inSpec, outSpec).apply(input);

        assertEquals(new Shape(4, 2), output.shape());
        float[] expected = new float[]{
                101f, 101f,
                102f, 102f,
                103f, 103f,
                104f, 104f
        };
        assertArrayEquals(expected, output.toFloatArray());
    }
}
