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
 * Tests for the 2-D cuFFT bridge — {@code fft2OnDevice} and the auto-dispatch
 * inside the public {@code fft2}/{@code ifft2} entry points, which now offload
 * even <em>host-resident</em> inputs by falling back to the default GPU device.
 * Correctness is checked against an independent naive O(N²) DFT, not against
 * {@code Fft.fft2(host)} (which itself auto-offloads once a GPU is present).
 * Every test skips unless a real GPU + cuFFT is available, matching
 * {@link FftGpuTest} / {@link FftGpu3DTest}.
 */
public class FftGpu2DTest {

    private static final double TOL_F64 = 1e-7;   // cuFFT vs naive DFT, these sizes
    private static final double TOL_F32 = 1e-3;   // relative, single precision C2C

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
            "Default device is not a GPU; skipping cuFFT 2-D test");
        try {
            Class.forName("uk.ac.manchester.tornado.cufft.CuFft");
        } catch (Throwable t) {
            Assumptions.abort("tornado-cufft not on classpath: " + t.getMessage());
        }
        return dev;
    }

    /** Independent oracle: direct O(N²) 2-D DFT, numpy convention (1/N on inverse). */
    private static double[][] naiveDft2(double[] re, double[] im, int n0, int n1, boolean inverse) {
        int m = n0 * n1;
        double[] outRe = new double[m], outIm = new double[m];
        double sign = inverse ? 1.0 : -1.0;
        double scale = inverse ? 1.0 / m : 1.0;
        for (int k0 = 0; k0 < n0; k0++) {
            for (int k1 = 0; k1 < n1; k1++) {
                double sr = 0, si = 0;
                for (int j0 = 0; j0 < n0; j0++) {
                    for (int j1 = 0; j1 < n1; j1++) {
                        double ang = sign * 2 * Math.PI * ((double) k0 * j0 / n0 + (double) k1 * j1 / n1);
                        double c = Math.cos(ang), s = Math.sin(ang);
                        int ji = j0 * n1 + j1;
                        sr += re[ji] * c - im[ji] * s;
                        si += re[ji] * s + im[ji] * c;
                    }
                }
                int ki = k0 * n1 + k1;
                outRe[ki] = sr * scale;
                outIm[ki] = si * scale;
            }
        }
        return new double[][] {outRe, outIm};
    }

    private static void assertOnGpu(NDArray a, String label) {
        Device d = a.device();
        TornadoDevice td = (d == null) ? null : d.getTornadoDevice();
        if (td == null || td.getDeviceType() != TornadoDeviceType.GPU) {
            throw new AssertionError(label + " did not land on a GPU device (was " + d + ")");
        }
    }

    private static void assertCloseAbs(double[] ref, double[] got, double tol, String label) {
        double maxAbs = 0.0;
        int worst = 0;
        for (int i = 0; i < ref.length; i++) {
            double dd = Math.abs(ref[i] - got[i]);
            if (dd > maxAbs) { maxAbs = dd; worst = i; }
        }
        if (maxAbs > tol) {
            throw new AssertionError(label + ": max |Δ|=" + maxAbs + " at " + worst +
                " (ref=" + ref[worst] + " got=" + got[worst] + ")");
        }
    }

    private static void assertCloseRel(double[] ref, float[] got, double relTol, String label) {
        double maxRel = 0.0;
        int worst = 0;
        for (int i = 0; i < ref.length; i++) {
            double rel = Math.abs(ref[i] - got[i]) / (1.0 + Math.abs(ref[i]));
            if (rel > maxRel) { maxRel = rel; worst = i; }
        }
        if (maxRel > relTol) {
            throw new AssertionError(label + ": max rel Δ=" + maxRel + " at " + worst +
                " (ref=" + ref[worst] + " got=" + got[worst] + ")");
        }
    }

    private static double[] randomVector(int n, long seed) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextDouble() * 2 - 1;
        return x;
    }

    /** Explicit device path (FP64) matches the naive DFT, forward. */
    @Test
    public void gpuFft2F64MatchesNaiveDft() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 8, n1 = 16;
        double[] hre = randomVector(n0 * n1, 1L);
        double[] him = randomVector(n0 * n1, 2L);
        Shape sh = new Shape(n0, n1);

        double[][] ref = naiveDft2(hre, him, n0, n1, /*inverse=*/false);
        NDArray[] gpu = Fft.fft2OnDevice(
            new ConcreteNDArray(hre.clone(), sh, dev),
            new ConcreteNDArray(him.clone(), sh, dev),
            false);
        assertOnGpu(gpu[0], "fft2OnDevice");
        assertCloseAbs(ref[0], gpu[0].toDoubleArray(), TOL_F64, "fft2 re");
        assertCloseAbs(ref[1], gpu[1].toDoubleArray(), TOL_F64, "fft2 im");
    }

    /** Explicit device path (FP32 C2C) matches the naive DFT, forward. */
    @Test
    public void gpuFft2F32MatchesNaiveDft() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 16, n1 = 16;
        double[] hre = randomVector(n0 * n1, 3L);
        double[] him = randomVector(n0 * n1, 4L);
        float[] fre = new float[hre.length], fim = new float[him.length];
        for (int i = 0; i < hre.length; i++) { fre[i] = (float) hre[i]; fim[i] = (float) him[i]; }
        Shape sh = new Shape(n0, n1);

        double[][] ref = naiveDft2(hre, him, n0, n1, false);
        NDArray[] gpu = Fft.fft2OnDevice(
            new ConcreteNDArray(fre, sh, DType.FLOAT32, dev),
            new ConcreteNDArray(fim, sh, DType.FLOAT32, dev),
            false);
        assertOnGpu(gpu[0], "fft2OnDevice(F32)");
        assertCloseRel(ref[0], gpu[0].toFloatArray(), TOL_F32, "fft2 re (F32)");
        assertCloseRel(ref[1], gpu[1].toFloatArray(), TOL_F32, "fft2 im (F32)");
    }

    /** ifft2 ∘ fft2 on the device recovers the input (guards the single 1/N). */
    @Test
    public void gpuIfft2ReversesFft2() {
        Device dev = gpuDeviceOrSkip();
        int n0 = 32, n1 = 8;
        double[] hre = randomVector(n0 * n1, 5L);
        double[] him = randomVector(n0 * n1, 6L);
        Shape sh = new Shape(n0, n1);

        NDArray[] fwd = Fft.fft2OnDevice(
            new ConcreteNDArray(hre.clone(), sh, dev),
            new ConcreteNDArray(him.clone(), sh, dev), false);
        NDArray[] back = Fft.fft2OnDevice(fwd[0], fwd[1], /*inverse=*/true);

        assertCloseAbs(hre, back[0].toDoubleArray(), TOL_F64, "fft2->ifft2 re");
        assertCloseAbs(him, back[1].toDoubleArray(), TOL_F64, "fft2->ifft2 im");
    }

    /**
     * Auto-dispatch: {@link Fft#fft2} on <em>host</em> inputs must silently
     * offload to the GPU (the 2-D examples — Ginzburg–Landau, Navier–Stokes —
     * keep their state as host NDArrays). Asserts both that the result lands on
     * a GPU device and that it matches the naive DFT.
     */
    @Test
    public void autoDispatchFft2FromHostRoutesToGpu() {
        gpuDeviceOrSkip();
        int n0 = 8, n1 = 16;
        double[] hre = randomVector(n0 * n1, 42L);
        double[] him = randomVector(n0 * n1, 43L);
        Shape sh = new Shape(n0, n1);

        double[][] ref = naiveDft2(hre, him, n0, n1, false);
        NDArray[] auto = Fft.fft2(
            new ConcreteNDArray(hre.clone(), sh),   // host-resident
            new ConcreteNDArray(him.clone(), sh));
        assertOnGpu(auto[0], "auto-dispatch fft2 from host");
        assertCloseAbs(ref[0], auto[0].toDoubleArray(), TOL_F64, "auto fft2 re");
        assertCloseAbs(ref[1], auto[1].toDoubleArray(), TOL_F64, "auto fft2 im");
    }
}
