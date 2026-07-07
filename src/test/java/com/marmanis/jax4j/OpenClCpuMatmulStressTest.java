package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stress-tests large matrix multiplication on a TornadoVM OpenCL device
 * backed by the CPU (e.g. POCL exposing all cores of the host CPU as an
 * OpenCL compute device), comparing it against {@link Device#defaultDevice()}
 * (a real GPU when TornadoVM discovers one, e.g. this sandbox's RTX 4060
 * under the PTX backend).
 *
 * <p>This demonstrates the benefit of TornadoVM's OpenCL backend even
 * without a GPU: dispatching to the multi-core CPU through OpenCL lets
 * TornadoVM parallelize the matmul's {@code @Parallel} loops across all
 * cores, instead of running the single-threaded loop {@code ConcreteNDArray}
 * uses on {@link Device#host()}.
 *
 * <p>The single-threaded host baseline is <em>not</em> run by default — at
 * matrix sizes large enough to be interesting it dominates the test's
 * runtime. Opt in with {@code -Djax4j.test.includeHostBaseline=true} to also
 * compare against it.
 *
 * <p>Skipped entirely (no assertions run) when no OpenCL CPU device is
 * discovered, so this stays green on sandboxes without an OpenCL CPU runtime
 * configured (e.g. no POCL/Intel OpenCL ICD installed).
 */
public class OpenClCpuMatmulStressTest {
    private static final Logger log = LoggerFactory.getLogger(OpenClCpuMatmulStressTest.class);

    private static NDArray randomMatrix(int n, Device device, long seed) {
        Random rnd = new Random(seed);
        float[] data = new float[n * n];
        for (int i = 0; i < data.length; i++) data[i] = rnd.nextFloat();
        return new ConcreteNDArray(data, new Shape(n, n)).to(device);
    }

    private static Device findOpenClCpuDevice() {
        for (Device d : Device.getDevices()) {
            if (d.getTornadoDevice() != null && d.getTornadoDevice().getDeviceType() == TornadoDeviceType.CPU) {
                return d;
            }
        }
        return null;
    }

    @Test
    public void testLargeMatmulMatchesGpuAndIsFasterThanSingleThreadedHost() {
        Device openClCpu = findOpenClCpuDevice();
        if (openClCpu == null) {
            log.info("No OpenCL CPU device discovered (e.g. no POCL/Intel OpenCL ICD installed); skipping.");
            return;
        }

        final int n = 2048;
        Device gpu = Device.defaultDevice();
        boolean hasGpu = gpu.getTornadoDevice() != null
            && gpu.getTornadoDevice().getDeviceType() == TornadoDeviceType.GPU;

        NDArray aCl = randomMatrix(n, openClCpu, 1);
        NDArray bCl = randomMatrix(n, openClCpu, 2);

        // Warm up: trigger TornadoVM's task-graph compilation so the timed run
        // measures steady state, not first-call overhead.
        aCl.dot(bCl);

        long clStart = System.nanoTime();
        NDArray clResult = aCl.dot(bCl);
        long clElapsedMs = (System.nanoTime() - clStart) / 1_000_000;

        if (hasGpu) {
            NDArray aGpu = aCl.to(gpu);
            NDArray bGpu = bCl.to(gpu);
            aGpu.dot(bGpu);

            long gpuStart = System.nanoTime();
            NDArray gpuResult = aGpu.dot(bGpu);
            long gpuElapsedMs = (System.nanoTime() - gpuStart) / 1_000_000;

            log.info("{}x{} matmul -- OpenCL CPU {}: {} ms, GPU {}: {} ms",
                n, n, openClCpu, clElapsedMs, gpu, gpuElapsedMs);

            assertArrayEquals(gpuResult.toFloatArray(), clResult.toFloatArray(), 1e-1f,
                "OpenCL CPU and GPU matmul results must match");
        } else {
            log.info("{}x{} matmul -- OpenCL CPU {}: {} ms (no GPU discovered to compare against)",
                n, n, openClCpu, clElapsedMs);
        }

        if (Boolean.getBoolean("jax4j.test.includeHostBaseline")) {
            NDArray aHost = aCl.to(Device.host());
            NDArray bHost = bCl.to(Device.host());
            aHost.dot(bHost);

            long hostStart = System.nanoTime();
            NDArray hostResult = aHost.dot(bHost);
            long hostElapsedMs = (System.nanoTime() - hostStart) / 1_000_000;

            log.info("{}x{} matmul -- single-threaded host: {} ms", n, n, hostElapsedMs);

            assertArrayEquals(hostResult.toFloatArray(), clResult.toFloatArray(), 1e-1f,
                "OpenCL CPU and single-threaded host matmul results must match");
            assertTrue(clElapsedMs < hostElapsedMs,
                "Expected OpenCL CPU (" + clElapsedMs + " ms) to outperform the single-threaded host loop ("
                    + hostElapsedMs + " ms) for a " + n + "x" + n + " matmul by exploiting multiple cores");
        }
    }
}
