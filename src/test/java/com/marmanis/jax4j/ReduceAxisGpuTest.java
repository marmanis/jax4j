package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies axis reductions (sum and mean) executed on GPU/TornadoVM devices
 * produce identical results to host execution.
 */
public class ReduceAxisGpuTest {

    private static NDArray matrix() {
        return new ConcreteNDArray(new float[]{1, 2, 3, 10, 20, 30}, new Shape(2, 3));
    }

    @Test
    public void testAxisReductionOnGpuDevice() {
        Device gpu = Device.getDevices().stream()
            .filter(d -> d.getTornadoDevice() != null)
            .findFirst()
            .orElse(null);

        if (gpu == null) {
            System.out.println("No TornadoVM device detected, skipping GPU axis reduction test.");
            return;
        }

        System.out.println("Running GPU axis reduction test on device: " + gpu);

        // Host reference
        NDArray hostSumLast = matrix().sum(1);
        NDArray hostSumFirst = matrix().sum(0);
        NDArray hostMeanLast = matrix().mean(1);

        // GPU execution
        NDArray gpuSumLast = matrix().to(gpu).sum(1);
        NDArray gpuSumFirst = matrix().to(gpu).sum(0);
        NDArray gpuMeanLast = matrix().to(gpu).mean(1);

        // Check shapes
        assertEquals(hostSumLast.shape(), gpuSumLast.shape());
        assertEquals(hostSumFirst.shape(), gpuSumFirst.shape());
        assertEquals(hostMeanLast.shape(), gpuMeanLast.shape());

        // Check values
        assertArrayEquals(hostSumLast.toFloatArray(), gpuSumLast.toFloatArray(), 1e-4f);
        assertArrayEquals(hostSumFirst.toFloatArray(), gpuSumFirst.toFloatArray(), 1e-4f);
        assertArrayEquals(hostMeanLast.toFloatArray(), gpuMeanLast.toFloatArray(), 1e-4f);
    }
}
