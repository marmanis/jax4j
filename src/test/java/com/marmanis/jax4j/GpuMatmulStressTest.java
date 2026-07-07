package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stress-tests large square matrix multiplication on whatever device
 * {@link Device#defaultDevice()} resolves to (a real GPU when TornadoVM
 * discovers one, e.g. this sandbox's RTX 4060 under the PTX backend; the host
 * CPU otherwise) and compares it against the same computation forced onto
 * {@link Device#host()}.
 *
 * <p>Correctness is asserted unconditionally. The speedup assertion is
 * skipped when no real accelerator is present (host vs host has no reason to
 * differ), so this test stays green in environments without a GPU while
 * still demonstrating the GPU's advantage whenever one is available.
 */
public class GpuMatmulStressTest {
    private static final Logger log = LoggerFactory.getLogger(GpuMatmulStressTest.class);

    private static NDArray randomMatrix(int n, Device device, long seed) {
        Random rnd = new Random(seed);
        float[] data = new float[n * n];
        for (int i = 0; i < data.length; i++) data[i] = rnd.nextFloat();
        return new ConcreteNDArray(data, new Shape(n, n)).to(device);
    }

    @Test
    public void testLargeMatmulMatchesHostAndIsFasterOnGpu() {
        final int n = 1536; // 1536x1536 @ 1536x1536 = ~7.2B FLOPs per matmul
        boolean hasAccelerator = !Device.getDevices().isEmpty();
        Device fast = Device.defaultDevice();

        NDArray aFast = randomMatrix(n, fast, 1);
        NDArray bFast = randomMatrix(n, fast, 2);
        NDArray aHost = aFast.to(Device.host());
        NDArray bHost = bFast.to(Device.host());

        // Warm up (JIT compile the host loop, and on the GPU path trigger
        // TornadoVM's task-graph compilation) so the timed run measures steady
        // state, not first-call overhead.
        aFast.dot(bFast);
        aHost.dot(bHost);

        long fastStart = System.nanoTime();
        NDArray fastResult = aFast.dot(bFast);
        long fastElapsedMs = (System.nanoTime() - fastStart) / 1_000_000;

        long hostStart = System.nanoTime();
        NDArray hostResult = aHost.dot(bHost);
        long hostElapsedMs = (System.nanoTime() - hostStart) / 1_000_000;

        log.info("{}x{} matmul -- {}: {} ms, host: {} ms", n, n, fast, fastElapsedMs, hostElapsedMs);

        assertArrayEquals(hostResult.toFloatArray(), fastResult.toFloatArray(), 1e-1f,
            "GPU and host matmul results must match");

        if (hasAccelerator) {
            assertTrue(fastElapsedMs < hostElapsedMs,
                "Expected GPU (" + fastElapsedMs + " ms) to outperform host (" + hostElapsedMs
                    + " ms) for a " + n + "x" + n + " matmul");
        } else {
            log.info("No accelerator discovered; skipping speedup assertion (ran host vs host).");
        }
    }
}
