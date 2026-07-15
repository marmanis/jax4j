package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * Discrete Fourier transforms, mirroring {@code jax.numpy.fft}. Host-only for
 * now (no TornadoVM kernel, no VJP/vmap rules — treated as a numerical
 * primitive rather than a traced op, same status as {@code CAST}/gather). Both
 * {@code FLOAT32} and {@code FLOAT64} inputs are supported; the output shares
 * the input's dtype.
 *
 * <p>The complex transforms ({@link #fft}/{@link #ifft}) implement an iterative
 * radix-2 Cooley-Tukey algorithm and require the input length to be a power of
 * two. {@link #dctI} — the real transform Chebyshev value/coefficient
 * conversions use — is built on the same FFT via the standard "mirror the
 * sequence to length {@code 2(N-1)}" trick and therefore requires
 * {@code N - 1} to be a power of two (i.e. {@code N ∈ {2, 3, 5, 9, 17, 33, ...}}),
 * matching the grid sizes chebfun's adaptive constructor tries.
 */
public final class Fft {
    private Fft() {}

    /**
     * Forward complex FFT: {@code y_k = sum_j (re_j + i*im_j) *
     * exp(-2*pi*i*j*k/N)}. Both inputs must be 1-D vectors of the same
     * floating dtype and power-of-two length.
     *
     * @return length-2 array {@code {real, imag}} of the transformed signal.
     */
    public static NDArray[] fft(NDArray re, NDArray im) {
        return complexTransform(re, im, false);
    }

    /**
     * Inverse complex FFT: {@code y_k = (1/N) sum_j (re_j + i*im_j) *
     * exp(+2*pi*i*j*k/N)}. Both inputs must be 1-D vectors of the same
     * floating dtype and power-of-two length.
     */
    public static NDArray[] ifft(NDArray re, NDArray im) {
        return complexTransform(re, im, true);
    }

    /**
     * Real-input forward FFT, mirroring {@code numpy.fft.rfft}. For a
     * length-{@code N} real signal, only the first {@code N/2 + 1} DFT
     * outputs are independent (the rest are complex conjugates by
     * Hermitian symmetry), so the return is a pair of arrays of length
     * {@code N/2 + 1}. Length {@code N} must be a power of two. Input must
     * be 1-D FLOAT32 or FLOAT64.
     *
     * <p>Implemented by promoting to a complex FFT with zero imaginary
     * part and slicing — half the theoretical work of a specialised real
     * transform but simple and cache-friendly; upgrade to a split-radix
     * real FFT if profiling ever demands it.
     */
    public static NDArray[] rfft(NDArray x) {
        if (x.shape().rank() != 1) {
            throw new IllegalArgumentException("rfft requires a 1-D input, got shape " + x.shape());
        }
        int n = x.shape().dimensions()[0];
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("rfft length must be a power of two, got " + n);
        }
        int half = n / 2 + 1;
        if (x.dtype() == DType.FLOAT64) {
            double[] re = x.toDoubleArray().clone();
            double[] im = new double[n];
            fftInPlaceDouble(re, im, false);
            return new NDArray[]{
                new ConcreteNDArray(java.util.Arrays.copyOf(re, half), new Shape(half)),
                new ConcreteNDArray(java.util.Arrays.copyOf(im, half), new Shape(half))
            };
        }
        if (x.dtype() == DType.FLOAT32) {
            float[] re = x.toFloatArray().clone();
            float[] im = new float[n];
            fftInPlaceFloat(re, im, false);
            return new NDArray[]{
                new ConcreteNDArray(java.util.Arrays.copyOf(re, half), new Shape(half)),
                new ConcreteNDArray(java.util.Arrays.copyOf(im, half), new Shape(half))
            };
        }
        throw new IllegalArgumentException("rfft requires FLOAT32 or FLOAT64 input, got " + x.dtype());
    }

    /**
     * Inverse real-input FFT, mirroring {@code numpy.fft.irfft(a, n=2*(len(a)-1))}.
     * Given the {@code N/2 + 1} independent DFT bins of a real signal of
     * length {@code N} (as returned by {@link #rfft}), reconstructs the
     * length-{@code N} real signal. {@code N} must be a power of two;
     * we recover it as {@code 2 * (re.length - 1)}.
     */
    public static NDArray irfft(NDArray re, NDArray im) {
        if (re.shape().rank() != 1 || im.shape().rank() != 1) {
            throw new IllegalArgumentException(
                "irfft requires 1-D vectors, got shapes " + re.shape() + ", " + im.shape());
        }
        int half = re.shape().dimensions()[0];
        if (half != im.shape().dimensions()[0]) {
            throw new IllegalArgumentException(
                "irfft real and imaginary parts must have the same length: " +
                half + " vs " + im.shape().dimensions()[0]);
        }
        if (re.dtype() != im.dtype()) {
            throw new IllegalArgumentException(
                "irfft real and imaginary parts must share dtype: " +
                re.dtype() + " vs " + im.dtype());
        }
        int n = 2 * (half - 1);
        if (n < 2 || Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException(
                "irfft input length implies N = " + n + ", which must be a power of two >= 2");
        }
        if (re.dtype() == DType.FLOAT64) {
            double[] halfRe = re.toDoubleArray();
            double[] halfIm = im.toDoubleArray();
            double[] fullRe = new double[n];
            double[] fullIm = new double[n];
            System.arraycopy(halfRe, 0, fullRe, 0, half);
            System.arraycopy(halfIm, 0, fullIm, 0, half);
            // Hermitian conjugate mirror: X[N-k] = conj(X[k]) for k = 1..N/2-1.
            for (int k = 1; k < half - 1; k++) {
                fullRe[n - k] = halfRe[k];
                fullIm[n - k] = -halfIm[k];
            }
            fftInPlaceDouble(fullRe, fullIm, true);
            // Imaginary residue is O(eps) — drop it.
            return new ConcreteNDArray(fullRe, new Shape(n));
        }
        if (re.dtype() == DType.FLOAT32) {
            float[] halfRe = re.toFloatArray();
            float[] halfIm = im.toFloatArray();
            float[] fullRe = new float[n];
            float[] fullIm = new float[n];
            System.arraycopy(halfRe, 0, fullRe, 0, half);
            System.arraycopy(halfIm, 0, fullIm, 0, half);
            for (int k = 1; k < half - 1; k++) {
                fullRe[n - k] = halfRe[k];
                fullIm[n - k] = -halfIm[k];
            }
            fftInPlaceFloat(fullRe, fullIm, true);
            return new ConcreteNDArray(fullRe, new Shape(n));
        }
        throw new IllegalArgumentException("irfft requires FLOAT32 or FLOAT64 input, got " + re.dtype());
    }

    /**
     * Discrete Cosine Transform of type I (unnormalized):
     * {@code y_k = x_0 + (-1)^k * x_{N-1} + 2 * sum_{j=1}^{N-2} x_j *
     * cos(pi*j*k/(N-1))} for {@code k = 0..N-1}. This is exactly the
     * evaluation-to-coefficients map (up to a straightforward endpoint /
     * length scaling) used by chebfun for values on Chebyshev-2nd-kind points
     * {@code x_j = cos(pi*j/(N-1))}. It is self-inverse up to the factor
     * {@code 2*(N-1)}.
     *
     * <p>Requires {@code N - 1} to be a power of two (or {@code N == 1}, which
     * returns {@code x} unchanged). Implemented via a length-{@code 2*(N-1)}
     * FFT of {@code x} mirrored about both endpoints: {@code w = [x_0, x_1,
     * ..., x_{N-1}, x_{N-2}, ..., x_1]}. The real-and-even symmetry of
     * {@code w} makes the FFT output real (imaginary parts cancel in exact
     * arithmetic; we discard the small rounding residual and keep the reals).
     */
    public static NDArray dctI(NDArray x) {
        if (x.shape().rank() != 1) {
            throw new IllegalArgumentException("dctI requires a 1-D input, got shape " + x.shape());
        }
        int n = x.shape().dimensions()[0];
        if (n == 1) return x;
        if (n < 2) throw new IllegalArgumentException("dctI requires length >= 1, got " + n);
        int m = n - 1;
        if (Integer.bitCount(m) != 1) {
            throw new IllegalArgumentException(
                "dctI requires (N - 1) to be a power of two, got N = " + n);
        }
        if (x.dtype() == DType.FLOAT64) {
            return new ConcreteNDArray(dctIDouble(x.toDoubleArray()), new Shape(n));
        }
        if (x.dtype() == DType.FLOAT32) {
            return new ConcreteNDArray(dctIFloat(x.toFloatArray()), new Shape(n));
        }
        throw new IllegalArgumentException("dctI requires FLOAT32 or FLOAT64 input, got " + x.dtype());
    }

    /**
     * Raw {@code double[]} overload of {@link #dctI(NDArray)} — same
     * transform, but skips the {@link NDArray} / {@link Shape} wrapping on
     * the way in and out. Intended for tight host-side loops that call
     * DCT-I many times per second and can't afford the per-call allocation
     * of {@code ConcreteNDArray} and {@code Shape} objects on both sides
     * (chebfun4j's adaptive Chebtech constructor is the canonical
     * consumer: it hits this on every grid-doubling probe).
     *
     * <p>The input array is <em>not</em> mutated. The return is always a
     * fresh {@code double[]} — even in the length-1 special case, which
     * returns {@code {x[0]}} rather than aliasing the caller's array.
     *
     * <p>Same length constraint as {@link #dctI(NDArray)}: {@code x.length}
     * must be {@code 1} or of the form {@code 2^k + 1}.
     */
    public static double[] dctIRaw(double[] x) {
        int n = x.length;
        if (n == 0) throw new IllegalArgumentException("dctIRaw: input must be non-empty");
        if (n == 1) return new double[]{x[0]};
        int m = n - 1;
        if (Integer.bitCount(m) != 1) {
            throw new IllegalArgumentException(
                "dctIRaw requires (N - 1) to be a power of two, got N = " + n);
        }
        return dctIDouble(x);
    }

    private static NDArray[] complexTransform(NDArray re, NDArray im, boolean inverse) {
        if (re.shape().rank() != 1 || im.shape().rank() != 1) {
            throw new IllegalArgumentException(
                "fft/ifft require 1-D vectors, got shapes " + re.shape() + ", " + im.shape());
        }
        int n = re.shape().dimensions()[0];
        if (n != im.shape().dimensions()[0]) {
            throw new IllegalArgumentException(
                "fft/ifft real and imaginary parts must have the same length: " +
                n + " vs " + im.shape().dimensions()[0]);
        }
        if (re.dtype() != im.dtype()) {
            throw new IllegalArgumentException(
                "fft/ifft real and imaginary parts must share dtype: " +
                re.dtype() + " vs " + im.dtype());
        }
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("fft/ifft length must be a power of two, got " + n);
        }
        if (re.dtype() == DType.FLOAT64) {
            double[] a = re.toDoubleArray().clone();
            double[] b = im.toDoubleArray().clone();
            fftInPlaceDouble(a, b, inverse);
            return new NDArray[]{
                new ConcreteNDArray(a, new Shape(n)),
                new ConcreteNDArray(b, new Shape(n))
            };
        }
        if (re.dtype() == DType.FLOAT32) {
            float[] a = re.toFloatArray().clone();
            float[] b = im.toFloatArray().clone();
            fftInPlaceFloat(a, b, inverse);
            return new NDArray[]{
                new ConcreteNDArray(a, new Shape(n)),
                new ConcreteNDArray(b, new Shape(n))
            };
        }
        throw new IllegalArgumentException("fft/ifft require FLOAT32 or FLOAT64 input, got " + re.dtype());
    }

    // Iterative radix-2 Cooley-Tukey. `inverse` flips the sign of the twiddle
    // exponent and divides the result by n. Length is a power of two, checked
    // by the caller.
    private static void fftInPlaceDouble(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        double sign = inverse ? +1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1) {
            double theta = sign * 2.0 * Math.PI / len;
            double wStepRe = Math.cos(theta);
            double wStepIm = Math.sin(theta);
            int half = len >> 1;
            for (int i = 0; i < n; i += len) {
                double wRe = 1.0, wIm = 0.0;
                for (int k = 0; k < half; k++) {
                    int a = i + k;
                    int b = a + half;
                    double tRe = wRe * re[b] - wIm * im[b];
                    double tIm = wRe * im[b] + wIm * re[b];
                    re[b] = re[a] - tRe;
                    im[b] = im[a] - tIm;
                    re[a] += tRe;
                    im[a] += tIm;
                    double nRe = wRe * wStepRe - wIm * wStepIm;
                    double nIm = wRe * wStepIm + wIm * wStepRe;
                    wRe = nRe; wIm = nIm;
                }
            }
        }
        if (inverse) {
            double inv = 1.0 / n;
            for (int i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
        }
    }

    private static void fftInPlaceFloat(float[] re, float[] im, boolean inverse) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                float tr = re[i]; re[i] = re[j]; re[j] = tr;
                float ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        double sign = inverse ? +1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1) {
            double theta = sign * 2.0 * Math.PI / len;
            double wStepReD = Math.cos(theta);
            double wStepImD = Math.sin(theta);
            int half = len >> 1;
            for (int i = 0; i < n; i += len) {
                double wRe = 1.0, wIm = 0.0;
                for (int k = 0; k < half; k++) {
                    int a = i + k;
                    int b = a + half;
                    double tRe = wRe * re[b] - wIm * im[b];
                    double tIm = wRe * im[b] + wIm * re[b];
                    re[b] = (float) (re[a] - tRe);
                    im[b] = (float) (im[a] - tIm);
                    re[a] = (float) (re[a] + tRe);
                    im[a] = (float) (im[a] + tIm);
                    double nRe = wRe * wStepReD - wIm * wStepImD;
                    double nIm = wRe * wStepImD + wIm * wStepReD;
                    wRe = nRe; wIm = nIm;
                }
            }
        }
        if (inverse) {
            float inv = 1f / n;
            for (int i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
        }
    }

    private static double[] dctIDouble(double[] x) {
        int n = x.length;
        int m = 2 * (n - 1);
        double[] re = new double[m];
        double[] im = new double[m];
        // Mirror: w = [x_0, x_1, ..., x_{n-1}, x_{n-2}, ..., x_1]
        for (int k = 0; k < n; k++) re[k] = x[k];
        for (int k = 1; k < n - 1; k++) re[m - k] = x[k];
        fftInPlaceDouble(re, im, false);
        double[] out = new double[n];
        System.arraycopy(re, 0, out, 0, n);
        return out;
    }

    private static float[] dctIFloat(float[] x) {
        int n = x.length;
        int m = 2 * (n - 1);
        // Use double buffers internally for numerical accuracy, then round back.
        double[] re = new double[m];
        double[] im = new double[m];
        for (int k = 0; k < n; k++) re[k] = x[k];
        for (int k = 1; k < n - 1; k++) re[m - k] = x[k];
        fftInPlaceDouble(re, im, false);
        float[] out = new float[n];
        for (int k = 0; k < n; k++) out[k] = (float) re[k];
        return out;
    }
}
