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

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * cuFFT bridge tests: skipped unless the JVM is running under a TornadoVM
 * SDK that surfaces a non-host device AND {@code uk.ac.manchester.tornado.cufft.CuFft}
 * loads (i.e. {@code tornado-cufft.jar} is on the classpath and
 * {@code libtornado-cufft.so} + {@code libcufft.so.11} resolve). Under a
 * plain build (or a {@code -opencl}-only SDK) every test degrades to an
 * assumeTrue skip so CI stays green.
 */
public class FftGpuTest {

    private static final double TOL_F64 = 1e-9;
    private static final float TOL_F32 = 1e-4f;

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
            "Default device is not a GPU; skipping cuFFT test");
        try {
            Class.forName("uk.ac.manchester.tornado.cufft.CuFft");
        } catch (Throwable t) {
            Assumptions.abort("tornado-cufft not on classpath: " + t.getMessage());
        }
        return dev;
    }

    private static NDArray f64(double[] xs, Device dev) {
        return new ConcreteNDArray(xs, new Shape(xs.length), dev);
    }

    private static NDArray f32(float[] xs, Device dev) {
        return new ConcreteNDArray(xs, new Shape(xs.length), DType.FLOAT32, dev);
    }

    private static NDArray f64Host(double[] xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    private static NDArray f32Host(float[] xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    /**
     * Compare the GPU forward transform against the host implementation on a
     * random signal at each of a few power-of-two sizes. cuFFT and our
     * radix-2 loop share the same normalization convention (none on forward,
     * 1/N on inverse), so the magnitudes should match to within FP32 noise.
     */
    @Test
    public void gpuFftF32MatchesHostRandomSignal() {
        Device dev = gpuDeviceOrSkip();
        Random rng = new Random(0xC0FFEEL);
        for (int n : new int[] {16, 256, 4096}) {
            float[] hre = new float[n];
            float[] him = new float[n];
            for (int i = 0; i < n; i++) {
                hre[i] = rng.nextFloat() * 2f - 1f;
                him[i] = rng.nextFloat() * 2f - 1f;
            }

            NDArray[] host = Fft.fft(f32Host(hre), f32Host(him));
            NDArray[] gpu  = Fft.fftOnDevice(f32(hre, dev), f32(him, dev), /*inverse=*/false);
            assertNotNull(gpu);

            float[] hR = host[0].toFloatArray(), hI = host[1].toFloatArray();
            float[] gR = gpu[0].toFloatArray(),  gI = gpu[1].toFloatArray();
            for (int i = 0; i < n; i++) {
                float dr = Math.abs(hR[i] - gR[i]);
                float di = Math.abs(hI[i] - gI[i]);
                if (dr > TOL_F32 || di > TOL_F32) {
                    throw new AssertionError(
                        "n=" + n + " bin " + i + ": host=(" + hR[i] + "," + hI[i] +
                        ") gpu=(" + gR[i] + "," + gI[i] + ") |Δ|=(" + dr + "," + di + ")");
                }
            }
        }
    }

    /**
     * FP64 (cuFFT Z2Z) parity check.
     */
    @Test
    public void gpuFftF64MatchesHostRandomSignal() {
        Device dev = gpuDeviceOrSkip();
        Random rng = new Random(0xBEEFL);
        for (int n : new int[] {16, 256, 4096}) {
            double[] hre = new double[n];
            double[] him = new double[n];
            for (int i = 0; i < n; i++) {
                hre[i] = rng.nextDouble() * 2 - 1;
                him[i] = rng.nextDouble() * 2 - 1;
            }

            NDArray[] host = Fft.fft(f64Host(hre), f64Host(him));
            NDArray[] gpu  = Fft.fftOnDevice(f64(hre, dev), f64(him, dev), false);
            assertNotNull(gpu);

            double[] hR = host[0].toDoubleArray(), hI = host[1].toDoubleArray();
            double[] gR = gpu[0].toDoubleArray(),  gI = gpu[1].toDoubleArray();
            for (int i = 0; i < n; i++) {
                double dr = Math.abs(hR[i] - gR[i]);
                double di = Math.abs(hI[i] - gI[i]);
                if (dr > TOL_F64 * n || di > TOL_F64 * n) {
                    throw new AssertionError(
                        "n=" + n + " bin " + i + ": host=(" + hR[i] + "," + hI[i] +
                        ") gpu=(" + gR[i] + "," + gI[i] + ") |Δ|=(" + dr + "," + di + ")");
                }
            }
        }
    }

    /**
     * fft ∘ ifft round-trip must recover the input, both directions issued
     * against the device. Guards the 1/N normalization the wrapper adds
     * (cuFFT itself doesn't).
     */
    @Test
    public void gpuRoundTripFP32() {
        Device dev = gpuDeviceOrSkip();
        int n = 1024;
        Random rng = new Random(0xDEADL);
        float[] hre = new float[n];
        float[] him = new float[n];
        for (int i = 0; i < n; i++) {
            hre[i] = rng.nextFloat();
            him[i] = rng.nextFloat();
        }

        NDArray[] fwd = Fft.fftOnDevice(f32(hre, dev), f32(him, dev), false);
        NDArray[] back = Fft.fftOnDevice(fwd[0], fwd[1], true);

        float[] rR = back[0].toFloatArray(), rI = back[1].toFloatArray();
        for (int i = 0; i < n; i++) {
            if (Math.abs(rR[i] - hre[i]) > TOL_F32 || Math.abs(rI[i] - him[i]) > TOL_F32) {
                throw new AssertionError(
                    "round-trip bin " + i + ": got (" + rR[i] + "," + rI[i] +
                    ") expected (" + hre[i] + "," + him[i] + ")");
            }
        }
    }

    /**
     * Smoke perf: a length-{@code 2^16} complex FP32 FFT should complete on
     * the GPU in under two seconds end-to-end (transfer + dispatch + copy
     * back). Not a strict benchmark — the assertion is loose enough that
     * WSL cold-start still passes; a real slowdown would blow through it.
     */
    @Test
    public void gpuSmokePerfBelowTwoSeconds() {
        Device dev = gpuDeviceOrSkip();
        int n = 1 << 16;
        float[] hre = new float[n];
        float[] him = new float[n];
        for (int i = 0; i < n; i++) {
            hre[i] = (float) Math.sin(i * 0.001);
            him[i] = 0f;
        }
        NDArray reD = f32(hre, dev);
        NDArray imD = f32(him, dev);

        // Warm-up dispatch (JIT + first-touch CUDA context).
        Fft.fftOnDevice(reD, imD, false);

        long t0 = System.nanoTime();
        NDArray[] out = Fft.fftOnDevice(reD, imD, false);
        long dtMs = (System.nanoTime() - t0) / 1_000_000L;

        assertNotNull(out);
        if (dtMs > 2000) {
            throw new AssertionError("cuFFT dispatch took " + dtMs + " ms, expected < 2000 ms");
        }
    }
}
