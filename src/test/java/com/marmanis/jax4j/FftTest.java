package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Correctness checks for the FFT primitive. We verify a few closed-form
 * transforms (a length-4 complex signal, an all-ones signal), the FFT ↔ IFFT
 * round-trip, and that the DCT-I built on top of the FFT matches its direct
 * O(N²) definition. Chebfun4j's value ↔ coefficient conversions rely on
 * every one of these behaving correctly.
 */
public class FftTest {

    private static NDArray vec(double... xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    private static NDArray vec(float... xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testFftLength4DoubleAgainstClosedForm() {
        // f = [1, 2, 3, 4], imag = 0. DFT_k = sum_j f_j exp(-2πi jk/4).
        // Closed form: F = [10, -2+2i, -2, -2-2i].
        NDArray re = vec(1.0, 2.0, 3.0, 4.0);
        NDArray im = vec(0.0, 0.0, 0.0, 0.0);
        NDArray[] out = Fft.fft(re, im);
        double[] gotRe = out[0].toDoubleArray();
        double[] gotIm = out[1].toDoubleArray();
        double[] wantRe = {10, -2, -2, -2};
        double[] wantIm = {0, +2, 0, -2};
        double tol = 1e-12;
        for (int k = 0; k < 4; k++) {
            assertClose(wantRe[k], gotRe[k], tol, "re[" + k + "]");
            assertClose(wantIm[k], gotIm[k], tol, "im[" + k + "]");
        }
    }

    @Test
    public void testFftAllOnesIsDelta() {
        // FFT of a constant signal is a delta at k=0 with amplitude n.
        int n = 16;
        double[] one = new double[n];
        double[] zero = new double[n];
        for (int i = 0; i < n; i++) one[i] = 1.0;
        NDArray re = new ConcreteNDArray(one, new Shape(n));
        NDArray im = new ConcreteNDArray(zero, new Shape(n));
        NDArray[] out = Fft.fft(re, im);
        double[] gotRe = out[0].toDoubleArray();
        double[] gotIm = out[1].toDoubleArray();
        assertClose(n, gotRe[0], 1e-10, "F[0]");
        for (int k = 1; k < n; k++) {
            assertClose(0, gotRe[k], 1e-10, "F[" + k + "] real");
            assertClose(0, gotIm[k], 1e-10, "F[" + k + "] imag");
        }
    }

    @Test
    public void testFftIfftRoundTripDouble() {
        int n = 64;
        double[] re = new double[n];
        double[] im = new double[n];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < n; i++) { re[i] = rng.nextGaussian(); im[i] = rng.nextGaussian(); }
        double[] reOrig = re.clone();
        double[] imOrig = im.clone();
        NDArray reN = new ConcreteNDArray(re, new Shape(n));
        NDArray imN = new ConcreteNDArray(im, new Shape(n));
        NDArray[] fwd = Fft.fft(reN, imN);
        NDArray[] back = Fft.ifft(fwd[0], fwd[1]);
        double[] backRe = back[0].toDoubleArray();
        double[] backIm = back[1].toDoubleArray();
        for (int i = 0; i < n; i++) {
            assertClose(reOrig[i], backRe[i], 1e-12, "re round trip [" + i + "]");
            assertClose(imOrig[i], backIm[i], 1e-12, "im round trip [" + i + "]");
        }
    }

    @Test
    public void testFftFloatRoundTrip() {
        int n = 32;
        float[] re = new float[n];
        float[] im = new float[n];
        java.util.Random rng = new java.util.Random(7);
        for (int i = 0; i < n; i++) { re[i] = (float) rng.nextGaussian(); im[i] = (float) rng.nextGaussian(); }
        float[] reOrig = re.clone();
        float[] imOrig = im.clone();
        NDArray[] fwd = Fft.fft(vec(re), vec(im));
        NDArray[] back = Fft.ifft(fwd[0], fwd[1]);
        float[] backRe = back[0].toFloatArray();
        float[] backIm = back[1].toFloatArray();
        for (int i = 0; i < n; i++) {
            assertClose(reOrig[i], backRe[i], 1e-4, "re round trip [" + i + "]");
            assertClose(imOrig[i], backIm[i], 1e-4, "im round trip [" + i + "]");
        }
    }

    @Test
    public void testFftLengthMustBePowerOfTwo() {
        NDArray re = vec(1.0, 2.0, 3.0);
        NDArray im = vec(0.0, 0.0, 0.0);
        assertThrows(IllegalArgumentException.class, () -> Fft.fft(re, im));
    }

    @Test
    public void testDctIMatchesDirectFormula() {
        // Direct definition: y_k = x_0 + (-1)^k x_{n-1}
        //                        + 2 * sum_{j=1}^{n-2} x_j * cos(pi j k / (n-1)).
        int n = 17; // n - 1 = 16 = 2^4, valid.
        double[] x = new double[n];
        java.util.Random rng = new java.util.Random(11);
        for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
        double[] want = new double[n];
        for (int k = 0; k < n; k++) {
            double s = x[0] + ((k % 2 == 0) ? x[n - 1] : -x[n - 1]);
            for (int j = 1; j < n - 1; j++) s += 2.0 * x[j] * Math.cos(Math.PI * j * k / (n - 1));
            want[k] = s;
        }
        NDArray input = new ConcreteNDArray(x, new Shape(n));
        double[] got = Fft.dctI(input).toDoubleArray();
        for (int k = 0; k < n; k++) {
            assertClose(want[k], got[k], 1e-9, "DCT-I[" + k + "]");
        }
    }

    @Test
    public void testDctISelfInverseUpToScaling() {
        // DCT-I ∘ DCT-I = 2(N-1) * identity.
        int n = 33; // n - 1 = 32.
        double[] x = new double[n];
        java.util.Random rng = new java.util.Random(3);
        for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
        NDArray input = new ConcreteNDArray(x, new Shape(n));
        NDArray once = Fft.dctI(input);
        double[] twice = Fft.dctI(once).toDoubleArray();
        double scale = 2.0 * (n - 1);
        for (int i = 0; i < n; i++) {
            assertClose(scale * x[i], twice[i], 1e-9, "self-inverse [" + i + "]");
        }
    }

    @Test
    public void testDctIRequiresValidLength() {
        // n = 6 -> n - 1 = 5, not a power of two.
        NDArray x = vec(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assertThrows(IllegalArgumentException.class, () -> Fft.dctI(x));
    }

    @Test
    public void testRfftMatchesFftOnRealInput() {
        int n = 32;
        double[] x = new double[n];
        double[] zeros = new double[n];
        java.util.Random rng = new java.util.Random(13);
        for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
        NDArray xArr = new ConcreteNDArray(x, new Shape(n));
        NDArray zArr = new ConcreteNDArray(zeros, new Shape(n));
        NDArray[] full = Fft.fft(xArr, zArr);
        NDArray[] half = Fft.rfft(xArr);
        int m = n / 2 + 1;
        assertEquals(m, half[0].shape().dimensions()[0], "rfft length");
        double[] fullRe = full[0].toDoubleArray();
        double[] fullIm = full[1].toDoubleArray();
        double[] halfRe = half[0].toDoubleArray();
        double[] halfIm = half[1].toDoubleArray();
        for (int k = 0; k < m; k++) {
            assertClose(fullRe[k], halfRe[k], 1e-12, "rfft.re[" + k + "]");
            assertClose(fullIm[k], halfIm[k], 1e-12, "rfft.im[" + k + "]");
        }
    }

    @Test
    public void testRfftIrfftRoundTrip() {
        int n = 64;
        double[] x = new double[n];
        java.util.Random rng = new java.util.Random(19);
        for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
        double[] orig = x.clone();
        NDArray[] freq = Fft.rfft(new ConcreteNDArray(x, new Shape(n)));
        NDArray back = Fft.irfft(freq[0], freq[1]);
        double[] backArr = back.toDoubleArray();
        assertEquals(n, backArr.length);
        for (int i = 0; i < n; i++) {
            assertClose(orig[i], backArr[i], 1e-12, "rfft round-trip [" + i + "]");
        }
    }

    @Test
    public void testRfftFloatRoundTrip() {
        int n = 32;
        float[] x = new float[n];
        java.util.Random rng = new java.util.Random(23);
        for (int i = 0; i < n; i++) x[i] = (float) rng.nextGaussian();
        float[] orig = x.clone();
        NDArray[] freq = Fft.rfft(vec(x));
        NDArray back = Fft.irfft(freq[0], freq[1]);
        float[] backArr = back.toFloatArray();
        for (int i = 0; i < n; i++) {
            assertClose(orig[i], backArr[i], 1e-4, "rfft float round-trip [" + i + "]");
        }
    }

    @Test
    public void testRfftLengthMustBePowerOfTwo() {
        NDArray x = vec(1.0, 2.0, 3.0);
        assertThrows(IllegalArgumentException.class, () -> Fft.rfft(x));
    }

    @Test
    public void testDctILengthOneIsIdentity() {
        NDArray x = vec(3.14);
        NDArray y = Fft.dctI(x);
        assertEquals(1, y.shape().dimensions()[0]);
        assertClose(3.14, y.toDoubleArray()[0], 1e-15, "length-1 dctI");
    }

    // -----------------------------------------------------------------
    // dctIRaw — raw-double[] overload used by chebfun4j's hot path
    // -----------------------------------------------------------------

    @Test
    public void testDctIRawMatchesNdArrayVersion() {
        // Bit-for-bit match at every representative grid size chebfun4j
        // actually hits (adaptive constructor probes 3, 5, 9, 17, 33, ..., 65537).
        int[] sizes = {3, 5, 9, 17, 33, 65, 129, 257};
        java.util.Random rng = new java.util.Random(0xC4EBF00D);
        for (int n : sizes) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
            double[] viaRaw = Fft.dctIRaw(x);
            double[] viaNd  = Fft.dctI(new ConcreteNDArray(x.clone(), new Shape(n))).toDoubleArray();
            for (int k = 0; k < n; k++) {
                if (viaRaw[k] != viaNd[k]) {
                    throw new AssertionError(
                        "dctIRaw vs dctI diverge at n=" + n + " k=" + k
                            + ": raw=" + viaRaw[k] + " nd=" + viaNd[k]);
                }
            }
        }
    }

    @Test
    public void testDctIRawDoesNotMutateInput() {
        int n = 33;
        double[] x = new double[n];
        java.util.Random rng = new java.util.Random(0xBADF00D);
        for (int i = 0; i < n; i++) x[i] = rng.nextGaussian();
        double[] snapshot = x.clone();
        double[] unused = Fft.dctIRaw(x);
        for (int i = 0; i < n; i++) {
            if (x[i] != snapshot[i]) {
                throw new AssertionError("dctIRaw mutated input at i=" + i);
            }
        }
        // Prevent dead-store elimination.
        if (unused.length != n) throw new AssertionError();
    }

    @Test
    public void testDctIRawLengthOneReturnsFreshCopy() {
        double[] x = {3.14};
        double[] y = Fft.dctIRaw(x);
        assertClose(3.14, y[0], 1e-15, "length-1 value");
        // Not aliased — mutating one must not affect the other.
        y[0] = 0.0;
        assertClose(3.14, x[0], 1e-15, "input not aliased");
    }

    @Test
    public void testDctIRawRejectsInvalidLength() {
        // n = 6 -> n - 1 = 5, not a power of two.
        assertThrows(IllegalArgumentException.class,
            () -> Fft.dctIRaw(new double[]{1, 2, 3, 4, 5, 6}));
        // Empty array rejected.
        assertThrows(IllegalArgumentException.class,
            () -> Fft.dctIRaw(new double[0]));
    }
}
