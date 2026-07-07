package com.marmanis.jax4j;

import com.marmanis.jax4j.api.MultiDevice;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MultiDevice#map}, the concurrent fan-out primitive a future
 * {@code pmap} would build on. Uses synthetic {@code Device(null)} instances
 * since this sandbox has no real accelerators — the point under test is the
 * fan-out/placement/collection logic, not real parallel hardware execution.
 */
public class MultiDeviceTest {

    private static NDArray vec(float... v) {
        return new ConcreteNDArray(v, new Shape(v.length));
    }

    @Test
    public void testMapRunsOneShardPerDevice() {
        Device d0 = new Device(null);
        Device d1 = new Device(null);
        NDArray[] shards = {vec(1, 2, 3), vec(10, 20, 30)};

        // fn must not introduce a third array defaulted to Device.host() — that
        // would mismatch against the explicitly-placed shard (by design: see
        // DevicePlacementTest#testMismatchedExplicitDevicesThrow).
        NDArray[] results = MultiDevice.map(List.of(d0, d1), shards, x -> x.mul(x));

        assertArrayEquals(new float[]{1, 4, 9}, results[0].toFloatArray(), 1e-6f);
        assertArrayEquals(new float[]{100, 400, 900}, results[1].toFloatArray(), 1e-6f);
    }

    @Test
    public void testShardsArePlacedOnTheirAssignedDevice() {
        Device d0 = new Device(null);
        Device d1 = new Device(null);
        NDArray[] shards = {vec(1), vec(2)};

        NDArray[] results = MultiDevice.map(List.of(d0, d1), shards, x -> x);

        assertSame(d0, results[0].device());
        assertSame(d1, results[1].device());
    }

    @Test
    public void testRunsConcurrentlyAcrossDevices() throws InterruptedException {
        Device d0 = new Device(null);
        Device d1 = new Device(null);
        AtomicInteger concurrentCount = new AtomicInteger(0);
        AtomicInteger maxObservedConcurrency = new AtomicInteger(0);

        MultiDevice.map(List.of(d0, d1), new NDArray[]{vec(1), vec(2)}, x -> {
            int current = concurrentCount.incrementAndGet();
            maxObservedConcurrency.updateAndGet(prev -> Math.max(prev, current));
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            concurrentCount.decrementAndGet();
            return x;
        });

        assertEquals(2, maxObservedConcurrency.get(), "Both tasks should have overlapped");
    }

    @Test
    public void testMismatchedDeviceAndShardCountsThrow() {
        Device d0 = new Device(null);
        assertThrows(IllegalArgumentException.class,
            () -> MultiDevice.map(List.of(d0), new NDArray[]{vec(1), vec(2)}, x -> x));
    }

    @Test
    public void testTaskFailureSurfacesAsRuntimeException() {
        Device d0 = new Device(null);
        assertThrows(RuntimeException.class, () -> MultiDevice.map(List.of(d0), new NDArray[]{vec(1)}, x -> {
            throw new IllegalStateException("boom");
        }));
    }
}
