package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import java.util.Random;

/**
 * Tests for the 3-D cuFFT bridge — {@code fft3OnDevice}, {@code rfft3OnDevice},
 * {@code irfft3OnDevice}, and the auto-dispatch inside the public
 * {@code fft3}/{@code rfft3}/{@code irfft3} entry points. Every test skips
 * unless a real GPU + cuFFT is available, matching the FftGpuTest pattern.
 */
public class FftGpu3DTest {

    private static final double TOL_F64 = 1e-8;
    private static final float TOL_F32 = 1e-3f;

    private static Device gpuDeviceOrSkip() {
        Device dev;
        try {
            dev = Device.defaultDevice();
        } catch (Throwable t) {
            Assumptions.abort("TornadoVM runtime not initialized: " + t.getMessage());
            return null;
        }
        TornadoDevice td = dev.getTornadoDevice();
        Assumptions.assumeTrue(td != null && td.getDeviceType() == TornadoDeviceType.GPU,
            "Default device is not a GPU; skipping cuFFT 3-D test");
        try {
            Class.forName("uk.ac.manchester.tornado.cufft.CuFft");
        } catch (Throwable t) {
            Assumptions.abort("tornado-cufft not on classpath: " + t.getMessage());
        }
        return dev;
    }

    private static double[] randomVector(int n, long seed) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextDouble() * 2 - 1;
        return x;
    }

    private static void assertMatches(double[] host, double[] gpu, double tol, String label) {
        double maxAbs = 0.0;
        int worst = 0;
        for (int i = 0; i < host.length; i++) {
            double d = Math.abs(host[i] - gpu[i]);
            if (d > maxAbs) { maxAbs = d; worst = i; }
        }
        if (maxAbs > tol) {
            throw new AssertionError(label + ": max |Δ|=" + maxAbs +
                " at index " + worst + " (host=" + host[worst] + " gpu=" + gpu[worst] + ")");
        }
    }

    @Test
    public void gpuFft3F64MatchesHost() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 16, n1 = 8, n2 = 32;
        int total = n0 * n1 * n2;
        double[] hre = randomVector(total, 1L);
        double[] him = randomVector(total, 2L);
        Shape sh = new Shape(n0, n1, n2);

        NDArray[] host = Fft.fft3(
            new ConcreteNDArray(hre.clone(), sh),
            new ConcreteNDArray(him.clone(), sh));
        NDArray[] gpu = Fft.fft3OnDevice(
            new ConcreteNDArray(hre.clone(), sh, dev),
            new ConcreteNDArray(him.clone(), sh, dev),
            /*inverse=*/false);

        assertMatches(host[0].toDoubleArray(), gpu[0].toDoubleArray(), TOL_F64 * total, "fft3 re");
        assertMatches(host[1].toDoubleArray(), gpu[1].toDoubleArray(), TOL_F64 * total, "fft3 im");
    }

    @Test
    public void gpuRfft3F64MatchesHost() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 16, n1 = 16, n2 = 32;
        int total = n0 * n1 * n2;
        double[] hx = randomVector(total, 7L);
        Shape sh = new Shape(n0, n1, n2);

        NDArray[] host = Fft.rfft3(new ConcreteNDArray(hx.clone(), sh));
        NDArray[] gpu = Fft.rfft3OnDevice(new ConcreteNDArray(hx.clone(), sh, dev));

        assertMatches(host[0].toDoubleArray(), gpu[0].toDoubleArray(), TOL_F64 * total, "rfft3 re");
        assertMatches(host[1].toDoubleArray(), gpu[1].toDoubleArray(), TOL_F64 * total, "rfft3 im");
    }

    @Test
    public void gpuIrfft3ReversesRfft3() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 8, n1 = 8, n2 = 16;
        int total = n0 * n1 * n2;
        double[] hx = randomVector(total, 11L);
        Shape sh = new Shape(n0, n1, n2);

        NDArray xDev = new ConcreteNDArray(hx.clone(), sh, dev);
        NDArray[] spec = Fft.rfft3OnDevice(xDev);
        NDArray back = Fft.irfft3OnDevice(spec[0], spec[1], n2);

        assertMatches(hx, back.toDoubleArray(), TOL_F64 * total, "rfft3 -> irfft3 round-trip");
    }

    /**
     * Auto-dispatch: {@link Fft#rfft3(NDArray)} should silently pick the
     * GPU path when its input is placed on a GPU device. We compare
     * against a host {@code rfft3} on the same data.
     */
    @Test
    public void autoDispatchRfft3RoutesToGpu() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 8, n1 = 8, n2 = 16;
        int total = n0 * n1 * n2;
        double[] hx = randomVector(total, 42L);
        Shape sh = new Shape(n0, n1, n2);

        NDArray[] host = Fft.rfft3(new ConcreteNDArray(hx.clone(), sh));
        NDArray[] auto = Fft.rfft3(new ConcreteNDArray(hx.clone(), sh, dev));

        assertMatches(host[0].toDoubleArray(), auto[0].toDoubleArray(), TOL_F64 * total,
            "auto-dispatch rfft3 re");
        assertMatches(host[1].toDoubleArray(), auto[1].toDoubleArray(), TOL_F64 * total,
            "auto-dispatch rfft3 im");
    }

    /**
     * Smoke perf: 64³ FP32 rfft3 in under 3 seconds. Loose bound —
     * meant to catch outright regressions, not micro-perf.
     */
    @Test
    public void gpuSmokePerf64CubeUnderThreeSeconds() {
        Device dev = gpuDeviceOrSkip();
        int n = 64;
        int total = n * n * n;
        float[] hx = new float[total];
        Random rng = new Random(0xF0F0L);
        for (int i = 0; i < total; i++) hx[i] = rng.nextFloat();
        Shape sh = new Shape(n, n, n);

        NDArray xDev = new ConcreteNDArray(hx, sh, DType.FLOAT32, dev);
        // Warm-up: JIT + cuFFT plan cache.
        Fft.rfft3OnDevice(xDev);

        long t0 = System.nanoTime();
        NDArray[] spec = Fft.rfft3OnDevice(xDev);
        long dtMs = (System.nanoTime() - t0) / 1_000_000L;

        if (spec == null || dtMs > 3000) {
            throw new AssertionError("rfft3OnDevice 64³ took " + dtMs + " ms, expected < 3000");
        }
    }
}
