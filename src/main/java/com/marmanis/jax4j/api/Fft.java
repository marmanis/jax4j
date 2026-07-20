package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.common.TornadoFunctions;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Discrete Fourier transforms, mirroring {@code jax.numpy.fft}. Two execution
 * paths, chosen automatically:
 *
 * <ul>
 *   <li><b>Host</b> — iterative radix-2 Cooley-Tukey in Java, both FP32 and
 *       FP64. Always available. Used when either input isn't on a
 *       TornadoVM device, or when the runtime doesn't ship cuFFT.</li>
 *   <li><b>NVIDIA cuFFT</b> — {@link #fft}/{@link #ifft} auto-dispatch to
 *       real cuFFT (via a TornadoVM {@code LibraryTaskDescriptor} pointing
 *       at {@code nvidia/cufft}) whenever both operands are placed on a
 *       non-host device and {@code libtornado-cufft.so} + {@code libcufft.so}
 *       are loadable. FP32 uses {@code cufftForwardC2C}; FP64 uses
 *       {@code cufftForwardZ2Z}. When cuFFT init fails at runtime (missing
 *       native lib, wrong SDK variant, etc.), the call transparently falls
 *       back to the host path so tests/examples keep working.</li>
 * </ul>
 *
 * <p>The complex transforms ({@link #fft}/{@link #ifft}) require the input
 * length to be a power of two. {@link #dctI} — the real transform Chebyshev
 * value/coefficient conversions use — is built on the same FFT via the
 * standard "mirror the sequence to length {@code 2(N-1)}" trick and
 * therefore requires {@code N - 1} to be a power of two (i.e.
 * {@code N ∈ {2, 3, 5, 9, 17, 33, ...}}), matching the grid sizes chebfun's
 * adaptive constructor tries. DCT-I stays on the host — the sizes chebfun4j
 * probes are small (n ≤ 257) and don't amortize a device round-trip.
 *
 * <p>No VJP or vmap rules (treated as a numerical primitive rather than a
 * traced op, same status as {@code CAST}/gather). Both {@code FLOAT32} and
 * {@code FLOAT64} inputs are supported; the output shares the input's
 * dtype.
 */
public final class Fft {
    private Fft() {}

    /**
     * Forward complex FFT: {@code y_k = sum_j (re_j + i*im_j) *
     * exp(-2*pi*i*j*k/N)}. Both inputs must be 1-D vectors of the same
     * floating dtype and power-of-two length. Auto-dispatches to NVIDIA
     * cuFFT if the inputs sit on a compatible device — see class Javadoc.
     *
     * @return length-2 array {@code {real, imag}} of the transformed signal.
     */
    public static NDArray[] fft(NDArray re, NDArray im) {
        NDArray[] gpu = tryCuFft(re, im, /*inverse=*/false);
        return (gpu != null) ? gpu : complexTransform(re, im, false);
    }

    /**
     * Inverse complex FFT: {@code y_k = (1/N) sum_j (re_j + i*im_j) *
     * exp(+2*pi*i*j*k/N)}. Both inputs must be 1-D vectors of the same
     * floating dtype and power-of-two length. Auto-dispatches to NVIDIA
     * cuFFT if the inputs sit on a compatible device.
     */
    public static NDArray[] ifft(NDArray re, NDArray im) {
        NDArray[] gpu = tryCuFft(re, im, /*inverse=*/true);
        return (gpu != null) ? gpu : complexTransform(re, im, true);
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

    /**
     * 3-D forward complex FFT. Transforms both real and imaginary inputs of shape (N1, N2, N3)
     * in-place along all three axes.
     */
    public static NDArray[] fft3(NDArray re, NDArray im) {
        NDArray[] gpu = tryCuFft3(re, im, false);
        return (gpu != null) ? gpu : complexTransform3D(re, im, false);
    }

    /**
     * 3-D inverse complex FFT. Transforms both real and imaginary inputs of shape (N1, N2, N3)
     * in-place along all three axes.
     */
    public static NDArray[] ifft3(NDArray re, NDArray im) {
        NDArray[] gpu = tryCuFft3(re, im, true);
        return (gpu != null) ? gpu : complexTransform3D(re, im, true);
    }

    /**
     * 3-D real forward FFT. Transforms a real input of shape (N1, N2, N3) into its
     * conjugate-symmetric complex frequency representation of shape (N1, N2, N3/2 + 1).
     */
    public static NDArray[] rfft3(NDArray x) {
        if (x.shape().rank() != 3) {
            throw new IllegalArgumentException("rfft3 requires a 3-D input, got shape " + x.shape());
        }
        int[] dims = x.shape().dimensions();
        int n0 = dims[0], n1 = dims[1], n2 = dims[2];
        if (Integer.bitCount(n0) != 1 || Integer.bitCount(n1) != 1 || Integer.bitCount(n2) != 1) {
            throw new IllegalArgumentException("rfft3 dimensions must be powers of two, got " + x.shape());
        }
        NDArray[] gpu = tryCuRfft3(x);
        if (gpu != null) return gpu;
        int half = n2 / 2 + 1;
        Shape outShape = new Shape(n0, n1, half);

        if (x.dtype() == DType.FLOAT64) {
            double[] src = x.toDoubleArray();
            double[] re = new double[n0 * n1 * half];
            double[] im = new double[n0 * n1 * half];
            double[] tempReal = new double[n2];
            double[] tempImag = new double[n2];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    int offset = (i * n1 + j) * n2;
                    System.arraycopy(src, offset, tempReal, 0, n2);
                    java.util.Arrays.fill(tempImag, 0.0);
                    fftInPlaceDouble(tempReal, tempImag, false);
                    int outOffset = (i * n1 + j) * half;
                    System.arraycopy(tempReal, 0, re, outOffset, half);
                    System.arraycopy(tempImag, 0, im, outOffset, half);
                }
            }
            transform3DDouble(re, im, n0, n1, half, 1, false);
            transform3DDouble(re, im, n0, n1, half, 0, false);
            return new NDArray[]{
                new ConcreteNDArray(re, outShape),
                new ConcreteNDArray(im, outShape)
            };
        }

        if (x.dtype() == DType.FLOAT32) {
            float[] src = x.toFloatArray();
            float[] re = new float[n0 * n1 * half];
            float[] im = new float[n0 * n1 * half];
            float[] tempReal = new float[n2];
            float[] tempImag = new float[n2];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    int offset = (i * n1 + j) * n2;
                    System.arraycopy(src, offset, tempReal, 0, n2);
                    java.util.Arrays.fill(tempImag, 0.0f);
                    fftInPlaceFloat(tempReal, tempImag, false);
                    int outOffset = (i * n1 + j) * half;
                    System.arraycopy(tempReal, 0, re, outOffset, half);
                    System.arraycopy(tempImag, 0, im, outOffset, half);
                }
            }
            transform3DFloat(re, im, n0, n1, half, 1, false);
            transform3DFloat(re, im, n0, n1, half, 0, false);
            return new NDArray[]{
                new ConcreteNDArray(re, outShape),
                new ConcreteNDArray(im, outShape)
            };
        }

        throw new IllegalArgumentException("rfft3 requires FLOAT32 or FLOAT64 input, got " + x.dtype());
    }

    /**
     * 3-D real inverse FFT. Reconstructs a real signal of shape (N1, N2, N3) from its
     * complex frequency representation of shape (N1, N2, N3/2 + 1).
     */
    public static NDArray irfft3(NDArray re, NDArray im) {
        if (re.shape().rank() != 3 || im.shape().rank() != 3) {
            throw new IllegalArgumentException("irfft3 requires 3-D inputs, got shapes " + re.shape() + ", " + im.shape());
        }
        if (!re.shape().equals(im.shape())) {
            throw new IllegalArgumentException("irfft3 real and imaginary parts must have the same shape, got " + re.shape() + ", " + im.shape());
        }
        if (re.dtype() != im.dtype()) {
            throw new IllegalArgumentException("irfft3 real and imaginary parts must share dtype: " + re.dtype() + " vs " + im.dtype());
        }
        int[] dims = re.shape().dimensions();
        int n0 = dims[0], n1 = dims[1], half = dims[2];
        int n2 = 2 * (half - 1);
        if (n2 < 2 || Integer.bitCount(n2) != 1) {
            throw new IllegalArgumentException("irfft3 input implies N3 = " + n2 + ", which must be a power of two >= 2");
        }
        NDArray gpu = tryCuIrfft3(re, im, n2);
        if (gpu != null) return gpu;
        Shape outShape = new Shape(n0, n1, n2);

        if (re.dtype() == DType.FLOAT64) {
            double[] workRe = re.toDoubleArray().clone();
            double[] workIm = im.toDoubleArray().clone();
            transform3DDouble(workRe, workIm, n0, n1, half, 0, true);
            transform3DDouble(workRe, workIm, n0, n1, half, 1, true);

            double[] out = new double[n0 * n1 * n2];
            double[] tempRe = new double[n2];
            double[] tempIm = new double[n2];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    int offset = (i * n1 + j) * half;
                    System.arraycopy(workRe, offset, tempRe, 0, half);
                    System.arraycopy(workIm, offset, tempIm, 0, half);
                    // Hermitian conjugate mirror
                    for (int k = 1; k < half - 1; k++) {
                        tempRe[n2 - k] = tempRe[k];
                        tempIm[n2 - k] = -tempIm[k];
                    }
                    fftInPlaceDouble(tempRe, tempIm, true);
                    int outOffset = (i * n1 + j) * n2;
                    System.arraycopy(tempRe, 0, out, outOffset, n2);
                }
            }
            return new ConcreteNDArray(out, outShape);
        }

        if (re.dtype() == DType.FLOAT32) {
            float[] workRe = re.toFloatArray().clone();
            float[] workIm = im.toFloatArray().clone();
            transform3DFloat(workRe, workIm, n0, n1, half, 0, true);
            transform3DFloat(workRe, workIm, n0, n1, half, 1, true);

            float[] out = new float[n0 * n1 * n2];
            float[] tempRe = new float[n2];
            float[] tempIm = new float[n2];
            for (int i = 0; i < n0; i++) {
                for (int j = 0; j < n1; j++) {
                    int offset = (i * n1 + j) * half;
                    System.arraycopy(workRe, offset, tempRe, 0, half);
                    System.arraycopy(workIm, offset, tempIm, 0, half);
                    // Hermitian conjugate mirror
                    for (int k = 1; k < half - 1; k++) {
                        tempRe[n2 - k] = tempRe[k];
                        tempIm[n2 - k] = -tempIm[k];
                    }
                    fftInPlaceFloat(tempRe, tempIm, true);
                    int outOffset = (i * n1 + j) * n2;
                    System.arraycopy(tempRe, 0, out, outOffset, n2);
                }
            }
            return new ConcreteNDArray(out, outShape);
        }

        throw new IllegalArgumentException("irfft3 requires FLOAT32 or FLOAT64 input, got " + re.dtype());
    }

    private static NDArray[] complexTransform3D(NDArray re, NDArray im, boolean inverse) {
        if (re.shape().rank() != 3 || im.shape().rank() != 3) {
            throw new IllegalArgumentException("fft3/ifft3 require 3-D inputs, got shapes " + re.shape() + ", " + im.shape());
        }
        if (!re.shape().equals(im.shape())) {
            throw new IllegalArgumentException("fft3/ifft3 real and imaginary parts must have the same shape, got " + re.shape() + ", " + im.shape());
        }
        if (re.dtype() != im.dtype()) {
            throw new IllegalArgumentException("fft3/ifft3 real and imaginary parts must share dtype: " + re.dtype() + " vs " + im.dtype());
        }
        int[] dims = re.shape().dimensions();
        int n0 = dims[0], n1 = dims[1], n2 = dims[2];
        if (Integer.bitCount(n0) != 1 || Integer.bitCount(n1) != 1 || Integer.bitCount(n2) != 1) {
            throw new IllegalArgumentException("fft3/ifft3 dimensions must be powers of two, got " + re.shape());
        }

        if (re.dtype() == DType.FLOAT64) {
            double[] a = re.toDoubleArray().clone();
            double[] b = im.toDoubleArray().clone();
            transform3DDouble(a, b, n0, n1, n2, 2, inverse);
            transform3DDouble(a, b, n0, n1, n2, 1, inverse);
            transform3DDouble(a, b, n0, n1, n2, 0, inverse);
            return new NDArray[]{
                new ConcreteNDArray(a, re.shape()),
                new ConcreteNDArray(b, im.shape())
            };
        }

        if (re.dtype() == DType.FLOAT32) {
            float[] a = re.toFloatArray().clone();
            float[] b = im.toFloatArray().clone();
            transform3DFloat(a, b, n0, n1, n2, 2, inverse);
            transform3DFloat(a, b, n0, n1, n2, 1, inverse);
            transform3DFloat(a, b, n0, n1, n2, 0, inverse);
            return new NDArray[]{
                new ConcreteNDArray(a, re.shape()),
                new ConcreteNDArray(b, im.shape())
            };
        }

        throw new IllegalArgumentException("fft3/ifft3 require FLOAT32 or FLOAT64 input, got " + re.dtype());
    }

    private static void transform3DDouble(double[] re, double[] im, int n0, int n1, int n2, int dim, boolean inverse) {
        if (dim == 0) {
            double[] tempRe = new double[n0];
            double[] tempIm = new double[n0];
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < n2; k++) {
                    int stride = n1 * n2;
                    for (int i = 0; i < n0; i++) {
                        int idx = i * stride + j * n2 + k;
                        tempRe[i] = re[idx];
                        tempIm[i] = im[idx];
                    }
                    fftInPlaceDouble(tempRe, tempIm, inverse);
                    for (int i = 0; i < n0; i++) {
                        int idx = i * stride + j * n2 + k;
                        re[idx] = tempRe[i];
                        im[idx] = tempIm[i];
                    }
                }
            }
        } else if (dim == 1) {
            double[] tempRe = new double[n1];
            double[] tempIm = new double[n1];
            for (int i = 0; i < n0; i++) {
                int stride = i * n1 * n2;
                for (int k = 0; k < n2; k++) {
                    for (int j = 0; j < n1; j++) {
                        int idx = stride + j * n2 + k;
                        tempRe[j] = re[idx];
                        tempIm[j] = im[idx];
                    }
                    fftInPlaceDouble(tempRe, tempIm, inverse);
                    for (int j = 0; j < n1; j++) {
                        int idx = stride + j * n2 + k;
                        re[idx] = tempRe[j];
                        im[idx] = tempIm[j];
                    }
                }
            }
        } else if (dim == 2) {
            double[] tempRe = new double[n2];
            double[] tempIm = new double[n2];
            for (int i = 0; i < n0; i++) {
                int stride = i * n1 * n2;
                for (int j = 0; j < n1; j++) {
                    int offset = stride + j * n2;
                    System.arraycopy(re, offset, tempRe, 0, n2);
                    System.arraycopy(im, offset, tempIm, 0, n2);
                    fftInPlaceDouble(tempRe, tempIm, inverse);
                    System.arraycopy(tempRe, 0, re, offset, n2);
                    System.arraycopy(tempIm, 0, im, offset, n2);
                }
            }
        }
    }

    private static void transform3DFloat(float[] re, float[] im, int n0, int n1, int n2, int dim, boolean inverse) {
        if (dim == 0) {
            float[] tempRe = new float[n0];
            float[] tempIm = new float[n0];
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < n2; k++) {
                    int stride = n1 * n2;
                    for (int i = 0; i < n0; i++) {
                        int idx = i * stride + j * n2 + k;
                        tempRe[i] = re[idx];
                        tempIm[i] = im[idx];
                    }
                    fftInPlaceFloat(tempRe, tempIm, inverse);
                    for (int i = 0; i < n0; i++) {
                        int idx = i * stride + j * n2 + k;
                        re[idx] = tempRe[i];
                        im[idx] = tempIm[i];
                    }
                }
            }
        } else if (dim == 1) {
            float[] tempRe = new float[n1];
            float[] tempIm = new float[n1];
            for (int i = 0; i < n0; i++) {
                int stride = i * n1 * n2;
                for (int k = 0; k < n2; k++) {
                    for (int j = 0; j < n1; j++) {
                        int idx = stride + j * n2 + k;
                        tempRe[j] = re[idx];
                        tempIm[j] = im[idx];
                    }
                    fftInPlaceFloat(tempRe, tempIm, inverse);
                    for (int j = 0; j < n1; j++) {
                        int idx = stride + j * n2 + k;
                        re[idx] = tempRe[j];
                        im[idx] = tempIm[j];
                    }
                }
            }
        } else if (dim == 2) {
            float[] tempRe = new float[n2];
            float[] tempIm = new float[n2];
            for (int i = 0; i < n0; i++) {
                int stride = i * n1 * n2;
                for (int j = 0; j < n1; j++) {
                    int offset = stride + j * n2;
                    System.arraycopy(re, offset, tempRe, 0, n2);
                    System.arraycopy(im, offset, tempIm, 0, n2);
                    fftInPlaceFloat(tempRe, tempIm, inverse);
                    System.arraycopy(tempRe, 0, re, offset, n2);
                    System.arraycopy(tempIm, 0, im, offset, n2);
                }
            }
        }
    }

    // ================================================================
    //  NVIDIA cuFFT bridge.
    //
    //  Design notes:
    //   * We construct a LibraryTaskDescriptor for "nvidia/cufft" directly
    //     using classes from tornado-api. The TornadoVM 5.x tornado-cufft
    //     jar is NOT on Maven Central, so depending on its typed CuFft
    //     helper would break every build that isn't a full CUDA install.
    //     LibraryTaskDescriptor is public API — we can build the same
    //     descriptor CuFft.cufftForwardC2C() would build.
    //   * Availability is probed once, lazily: we attempt to load the
    //     runtime type uk.ac.manchester.tornado.cufft.CuFft via reflection.
    //     If it isn't on the classpath the whole GPU path is disabled and
    //     the caller falls back to the host loop transparently. A single
    //     dispatch attempt that fails at runtime (e.g. libcufft.so.11 not
    //     installed) also disables the path for the remainder of the JVM
    //     so we don't repeat expensive JNI failures.
    //   * cuFFT does NOT normalize the inverse — the caller must divide
    //     by N. We do it on the host after unpacking; a 1/N kernel here
    //     would just add another dispatch for a tiny benefit.
    // ================================================================

    private static final String CUFFT_LIBRARY = "nvidia/cufft";
    private static final String CUFFT_FORWARD_C2C = "cufftForwardC2C";
    private static final String CUFFT_INVERSE_C2C = "cufftInverseC2C";
    private static final String CUFFT_FORWARD_Z2Z = "cufftForwardZ2Z";
    private static final String CUFFT_INVERSE_Z2Z = "cufftInverseZ2Z";

    // Volatile so a runtime disable in one thread is seen by all callers.
    private static volatile Boolean cufftAvailable = null;

    private static boolean cufftAvailable() {
        Boolean cached = cufftAvailable;
        if (cached != null) return cached;
        synchronized (Fft.class) {
            if (cufftAvailable != null) return cufftAvailable;
            // Probe by module presence, not by Class.forName: the cuFFT
            // package may not be exported to unnamed modules like jax4j,
            // but that doesn't stop TornadoVM's LibraryTask dispatch from
            // reaching the provider inside the module. The module simply
            // has to be present in the boot layer.
            boolean ok = ModuleLayer.boot().findModule("tornado.cufft").isPresent();
            String why = ok ? null : "module 'tornado.cufft' not in ModuleLayer.boot()";
            System.err.println(ok
                ? "[jax4j.Fft] cuFFT bridge active (tornado.cufft module resolved)"
                : "[jax4j.Fft] cuFFT bridge unavailable — using host loop. Reason: " + why);
            cufftAvailable = ok;
            return ok;
        }
    }

    /**
     * Attempt cuFFT dispatch. Returns {@code null} when the path is
     * unavailable — inputs on the host, non-power-of-two length, cuFFT
     * classes missing at runtime, or the dispatch itself throws. The
     * caller is responsible for taking the host fallback.
     */
    private static NDArray[] tryCuFft(NDArray re, NDArray im, boolean inverse) {
        if (!cufftAvailable()) return null;
        Device dev = re.device();
        if (dev == null || dev == Device.host()) return null;
        if (!dev.equals(im.device())) return null;
        if (re.dtype() != im.dtype()) return null;
        if (re.shape().rank() != 1 || im.shape().rank() != 1) return null;
        int n = re.shape().dimensions()[0];
        if (n < 2 || (n & (n - 1)) != 0) return null;
        try {
            return switch (re.dtype()) {
                case FLOAT32 -> cufftDispatchF32(re, im, dev, n, inverse);
                case FLOAT64 -> cufftDispatchF64(re, im, dev, n, inverse);
                default -> null;
            };
        } catch (Throwable t) {
            handleDispatchFailure("fft (1D)", t);
            return null;
        }
    }

    /**
     * Explicit cuFFT entry point. Throws {@link IllegalStateException}
     * when cuFFT is unavailable — no host fallback. Prefer
     * {@link #fft}/{@link #ifft}, which auto-dispatch and fall back.
     *
     * @param inverse forward transform if {@code false}, inverse (with
     *                {@code 1/N} scaling) if {@code true}
     */
    public static NDArray[] fftOnDevice(NDArray re, NDArray im, boolean inverse) {
        if (!cufftAvailable()) {
            throw new IllegalStateException(
                "cuFFT not on classpath: uk.ac.manchester.tornado.cufft.CuFft not loadable. "
                + "Use a TornadoVM -cuda or -full SDK and add tornado-cufft to the runtime classpath.");
        }
        Device dev = re.device();
        if (dev == null || dev == Device.host()) {
            throw new IllegalStateException("fftOnDevice requires inputs placed on a non-host Device");
        }
        int n = re.shape().dimensions()[0];
        NDArray[] out = switch (re.dtype()) {
            case FLOAT32 -> cufftDispatchF32(re, im, dev, n, inverse);
            case FLOAT64 -> cufftDispatchF64(re, im, dev, n, inverse);
            default -> throw new IllegalArgumentException(
                "fftOnDevice requires FLOAT32 or FLOAT64 inputs, got " + re.dtype());
        };
        if (out == null) {
            throw new IllegalStateException("cuFFT dispatch failed");
        }
        return out;
    }

    private static final Access[] CUFFT_ACCESS = {
        Access.READ_ONLY, Access.WRITE_ONLY, Access.READ_ONLY, Access.READ_ONLY
    };

    // The LibraryTask4 functional interface returns a LibraryTaskDescriptor
    // populated via fluent setters. Same shape as CuFft.cufftForward/InverseC2C
    // in the tornado-cufft jar — we just don't need that jar to construct it.
    private static TornadoFunctions.LibraryTask4<FloatArray, FloatArray, Integer, Integer>
            cufftC2C(boolean inverse) {
        String fn = inverse ? CUFFT_INVERSE_C2C : CUFFT_FORWARD_C2C;
        return (in, out, size, batch) -> new LibraryTaskDescriptor()
            .withLibrary(CUFFT_LIBRARY)
            .withFunction(fn)
            .withParameters(new Object[] { in, out, size, batch })
            .withAccess(CUFFT_ACCESS);
    }

    private static TornadoFunctions.LibraryTask4<DoubleArray, DoubleArray, Integer, Integer>
            cufftZ2Z(boolean inverse) {
        String fn = inverse ? CUFFT_INVERSE_Z2Z : CUFFT_FORWARD_Z2Z;
        return (in, out, size, batch) -> new LibraryTaskDescriptor()
            .withLibrary(CUFFT_LIBRARY)
            .withFunction(fn)
            .withParameters(new Object[] { in, out, size, batch })
            .withAccess(CUFFT_ACCESS);
    }

    private static NDArray[] cufftDispatchF32(
            NDArray re, NDArray im, Device dev, int n, boolean inverse) {
        float[] hre = re.toFloatArray();
        float[] him = im.toFloatArray();
        FloatArray input = new FloatArray(2 * n);
        FloatArray output = new FloatArray(2 * n);
        for (int i = 0; i < n; i++) {
            input.set(2 * i, hre[i]);
            input.set(2 * i + 1, him[i]);
        }

        TaskGraph graph = new TaskGraph("jax4j-fft-f32-" + Long.toHexString(System.nanoTime()))
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, input)
            .libraryTask("c2c", cufftC2C(inverse), input, output, n, 1)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot());
        try {
            plan.withDevice(dev.getTornadoDevice()).execute();
        } catch (Exception e) {
            throw new RuntimeException("cuFFT F32 dispatch failed", e);
        } finally {
            try { plan.freeDeviceMemory(); } catch (Throwable ignored) {}
            try { plan.close(); } catch (Throwable ignored) {}
        }

        float[] outRe = new float[n];
        float[] outIm = new float[n];
        if (inverse) {
            float inv = 1f / n;
            for (int i = 0; i < n; i++) {
                outRe[i] = output.get(2 * i) * inv;
                outIm[i] = output.get(2 * i + 1) * inv;
            }
        } else {
            for (int i = 0; i < n; i++) {
                outRe[i] = output.get(2 * i);
                outIm[i] = output.get(2 * i + 1);
            }
        }
        Shape sh = re.shape();
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, DType.FLOAT32, dev),
            new ConcreteNDArray(outIm, sh, DType.FLOAT32, dev)
        };
    }

    private static NDArray[] cufftDispatchF64(
            NDArray re, NDArray im, Device dev, int n, boolean inverse) {
        double[] hre = re.toDoubleArray();
        double[] him = im.toDoubleArray();
        DoubleArray input = new DoubleArray(2 * n);
        DoubleArray output = new DoubleArray(2 * n);
        for (int i = 0; i < n; i++) {
            input.set(2 * i, hre[i]);
            input.set(2 * i + 1, him[i]);
        }

        TaskGraph graph = new TaskGraph("jax4j-fft-f64-" + Long.toHexString(System.nanoTime()))
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, input)
            .libraryTask("z2z", cufftZ2Z(inverse), input, output, n, 1)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot());
        try {
            plan.withDevice(dev.getTornadoDevice()).execute();
        } catch (Exception e) {
            throw new RuntimeException("cuFFT F64 dispatch failed", e);
        } finally {
            try { plan.freeDeviceMemory(); } catch (Throwable ignored) {}
            try { plan.close(); } catch (Throwable ignored) {}
        }

        double[] outRe = new double[n];
        double[] outIm = new double[n];
        if (inverse) {
            double inv = 1.0 / n;
            for (int i = 0; i < n; i++) {
                outRe[i] = output.get(2 * i) * inv;
                outIm[i] = output.get(2 * i + 1) * inv;
            }
        } else {
            for (int i = 0; i < n; i++) {
                outRe[i] = output.get(2 * i);
                outIm[i] = output.get(2 * i + 1);
            }
        }
        Shape sh = re.shape();
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, dev),
            new ConcreteNDArray(outIm, sh, dev)
        };
    }

    // ================================================================
    //  3-D cuFFT bridge — device-resident, batched, single TaskGraph.
    //
    //  Strategy:
    //   * Represent a complex 3-D grid of shape (N0, N1, N2) as an
    //     interleaved FloatArray of length 2·N0·N1·N2, laid out
    //     row-major (last axis fastest — matches jax4j and NumPy).
    //   * cuFFT's batched 1-D plan only runs along the contiguous axis.
    //     To transform all three axes we rotate the layout twice between
    //     dispatches with a cyclic-shift kernel: (A,B,C) → (C,A,B).
    //     After three rotations the layout returns to (N0, N1, N2), so
    //     no separate "un-transpose" pass is needed.
    //   * The whole sequence — upload, 3 cuFFT calls, 3 kernel transposes,
    //     download — is one TaskGraph. TornadoVM keeps the two working
    //     buffers device-resident across the whole plan; the host only
    //     sees the packed input and the final result.
    //   * Two ping-pong buffers ({@code bufA}, {@code bufB}) are needed
    //     because cuFFT and the transpose kernel both require distinct
    //     input/output.
    // ================================================================

    /**
     * Cyclic-shift kernel: reads interleaved complex data laid out as
     * (D0, D1, D2) with the last axis contiguous, writes it out as
     * (D2, D0, D1). Each complex element is a pair of consecutive floats.
     *
     * <p>Public because TornadoVM's JIT needs to reach the method
     * reflectively when it compiles the task; keeping it package-private
     * is fine but public matches the pattern used elsewhere in jax4j.
     */
    public static void cyclicShiftF32(FloatArray in, FloatArray out, int D0, int D1, int D2) {
        for (@Parallel int k = 0; k < D2; k++) {
            for (@Parallel int i = 0; i < D0; i++) {
                for (@Parallel int j = 0; j < D1; j++) {
                    int inIdx = 2 * ((i * D1 + j) * D2 + k);
                    int outIdx = 2 * ((k * D0 + i) * D1 + j);
                    out.set(outIdx, in.get(inIdx));
                    out.set(outIdx + 1, in.get(inIdx + 1));
                }
            }
        }
    }

    public static void cyclicShiftF64(DoubleArray in, DoubleArray out, int D0, int D1, int D2) {
        for (@Parallel int k = 0; k < D2; k++) {
            for (@Parallel int i = 0; i < D0; i++) {
                for (@Parallel int j = 0; j < D1; j++) {
                    int inIdx = 2 * ((i * D1 + j) * D2 + k);
                    int outIdx = 2 * ((k * D0 + i) * D1 + j);
                    out.set(outIdx, in.get(inIdx));
                    out.set(outIdx + 1, in.get(inIdx + 1));
                }
            }
        }
    }

    /**
     * Full 3-D complex FFT on the GPU. Both inputs must be 3-D, same
     * shape, same non-host device, and each axis a power of two.
     * Auto-dispatched from {@link #fft3}/{@link #ifft3} when those
     * conditions hold; call directly for explicit control.
     */
    public static NDArray[] fft3OnDevice(NDArray re, NDArray im, boolean inverse) {
        if (!cufftAvailable()) {
            throw new IllegalStateException("cuFFT not on classpath");
        }
        Device dev = re.device();
        if (dev == null || dev == Device.host()) {
            throw new IllegalStateException("fft3OnDevice requires inputs placed on a non-host Device");
        }
        int[] d = re.shape().dimensions();
        if (d.length != 3) throw new IllegalArgumentException("fft3OnDevice requires a 3-D input");
        int n0 = d[0], n1 = d[1], n2 = d[2];
        return switch (re.dtype()) {
            case FLOAT32 -> cufftDispatch3DF32(re, im, dev, n0, n1, n2, inverse);
            case FLOAT64 -> cufftDispatch3DF64(re, im, dev, n0, n1, n2, inverse);
            default -> throw new IllegalArgumentException(
                "fft3OnDevice requires FLOAT32 or FLOAT64 inputs, got " + re.dtype());
        };
    }

    /**
     * Real 3-D FFT on the GPU. The input is a real N0×N1×N2 grid; the
     * output is its complex conjugate-symmetric spectrum, returned in
     * the same (N0, N1, N2/2+1) layout as the host {@link #rfft3}.
     * Internally does a full C2C transform and drops the redundant
     * half at unpack — the wasted work is amortized by the 3× axis
     * fanout of a batched cuFFT plan and by the single H↔D pair.
     */
    public static NDArray[] rfft3OnDevice(NDArray x) {
        if (!cufftAvailable()) throw new IllegalStateException("cuFFT not on classpath");
        Device dev = x.device();
        if (dev == null || dev == Device.host()) {
            throw new IllegalStateException("rfft3OnDevice requires input placed on a non-host Device");
        }
        int[] d = x.shape().dimensions();
        if (d.length != 3) throw new IllegalArgumentException("rfft3OnDevice requires a 3-D input");
        int n0 = d[0], n1 = d[1], n2 = d[2];
        return switch (x.dtype()) {
            case FLOAT32 -> rfft3DispatchF32(x, dev, n0, n1, n2);
            case FLOAT64 -> rfft3DispatchF64(x, dev, n0, n1, n2);
            default -> throw new IllegalArgumentException(
                "rfft3OnDevice requires FLOAT32 or FLOAT64 input, got " + x.dtype());
        };
    }

    /**
     * Inverse of {@link #rfft3OnDevice}. Reconstructs the redundant
     * conjugate half on the host, runs a full C2C inverse on the GPU,
     * and returns the real part.
     *
     * @param fullN2 the full (unhalved) length along the last axis, so
     *               the caller doesn't have to remember whether the
     *               spectrum was produced from an even- or odd-length
     *               signal. Practically always the source {@code N2}.
     */
    public static NDArray irfft3OnDevice(NDArray re, NDArray im, int fullN2) {
        if (!cufftAvailable()) throw new IllegalStateException("cuFFT not on classpath");
        Device dev = re.device();
        if (dev == null || dev == Device.host()) {
            throw new IllegalStateException("irfft3OnDevice requires input placed on a non-host Device");
        }
        int[] d = re.shape().dimensions();
        if (d.length != 3) throw new IllegalArgumentException("irfft3OnDevice requires a 3-D input");
        int n0 = d[0], n1 = d[1];
        int half = d[2];
        if (half != fullN2 / 2 + 1) {
            throw new IllegalArgumentException(
                "spectrum last-axis length " + half + " inconsistent with fullN2=" + fullN2);
        }
        return switch (re.dtype()) {
            case FLOAT32 -> irfft3DispatchF32(re, im, dev, n0, n1, fullN2);
            case FLOAT64 -> irfft3DispatchF64(re, im, dev, n0, n1, fullN2);
            default -> throw new IllegalArgumentException(
                "irfft3OnDevice requires FLOAT32 or FLOAT64 input, got " + re.dtype());
        };
    }

    private static NDArray[] cufftDispatch3DF32(
            NDArray re, NDArray im, Device dev, int n0, int n1, int n2, boolean inverse) {
        float[] hre = re.toFloatArray();
        float[] him = im.toFloatArray();
        int total = n0 * n1 * n2;
        Plan3F32 pp = acquireF32(n0, n1, n2, inverse, dev);
        FloatArray bufA = pp.bufA();
        for (int i = 0; i < total; i++) {
            bufA.set(2 * i, hre[i]);
            bufA.set(2 * i + 1, him[i]);
        }
        execPlan(pp.plan(), "cuFFT F32 3-D dispatch failed");

        float[] outRe = new float[total];
        float[] outIm = new float[total];
        float scale = inverse ? 1f / (float) total : 1f;
        for (int i = 0; i < total; i++) {
            outRe[i] = bufA.get(2 * i) * scale;
            outIm[i] = bufA.get(2 * i + 1) * scale;
        }
        Shape sh = re.shape();
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, DType.FLOAT32, dev),
            new ConcreteNDArray(outIm, sh, DType.FLOAT32, dev)
        };
    }

    private static NDArray[] cufftDispatch3DF64(
            NDArray re, NDArray im, Device dev, int n0, int n1, int n2, boolean inverse) {
        double[] hre = re.toDoubleArray();
        double[] him = im.toDoubleArray();
        int total = n0 * n1 * n2;
        Plan3F64 pp = acquireF64(n0, n1, n2, inverse, dev);
        DoubleArray bufA = pp.bufA();
        for (int i = 0; i < total; i++) {
            bufA.set(2 * i, hre[i]);
            bufA.set(2 * i + 1, him[i]);
        }
        execPlan(pp.plan(), "cuFFT F64 3-D dispatch failed");

        double[] outRe = new double[total];
        double[] outIm = new double[total];
        double scale = inverse ? 1.0 / total : 1.0;
        for (int i = 0; i < total; i++) {
            outRe[i] = bufA.get(2 * i) * scale;
            outIm[i] = bufA.get(2 * i + 1) * scale;
        }
        Shape sh = re.shape();
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, dev),
            new ConcreteNDArray(outIm, sh, dev)
        };
    }

    private static void execPlan(TornadoExecutionPlan p, String failMsg) {
        try {
            p.execute();
        } catch (Exception e) {
            throw new RuntimeException(failMsg, e);
        }
    }

    /**
     * Runs the three cuFFT dispatches + three transposes as one TaskGraph.
     * Final result lands in {@code bufA} in the original (n0,n1,n2) layout.
     *
     * <p>Ping-pong is deterministic: after three cyclic shifts we come
     * back to the start. Buffers alternate A→B→A→B→A→B→A.
     */
    // Cached execution plans (and their bound buffers) keyed by
    // (dtype, shape, direction, device). A fresh dispatch would create a
    // new cuFFT plan under the hood — cuFFT's own workspace, allocated
    // by libtornado-cufft.so, is not reclaimed by TornadoVM's
    // freeDeviceMemory(), so an uncached loop leaks per call and OOMs
    // within a handful of iterations at NS3D scales. Reusing one plan
    // per shape keeps cuFFT's workspace at exactly its cold-plan size.
    private record Plan3F32(FloatArray bufA, FloatArray bufB, TornadoExecutionPlan plan) {}
    private record Plan3F64(DoubleArray bufA, DoubleArray bufB, TornadoExecutionPlan plan) {}
    private static final ConcurrentHashMap<String, Plan3F32> plan3F32Cache = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Plan3F64> plan3F64Cache = new ConcurrentHashMap<>();

    private static String planKey(int n0, int n1, int n2, boolean inverse, Device dev) {
        return n0 + "x" + n1 + "x" + n2 + "-" + (inverse ? "inv" : "fwd")
            + "-" + System.identityHashCode(dev);
    }

    private static Plan3F32 acquireF32(int n0, int n1, int n2, boolean inverse, Device dev) {
        return plan3F32Cache.computeIfAbsent(planKey(n0, n1, n2, inverse, dev), k -> {
            int total = n0 * n1 * n2;
            FloatArray a = new FloatArray(2 * total);
            FloatArray b = new FloatArray(2 * total);
            TornadoFunctions.LibraryTask4<FloatArray, FloatArray, Integer, Integer> lp = cufftC2C(inverse);
            TaskGraph g = new TaskGraph("jax4j-fft3-f32-" + k)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a)
                .libraryTask("c2c-2", lp, a, b, n2, n0 * n1)
                .task("shift-1", Fft::cyclicShiftF32, b, a, n0, n1, n2)
                .libraryTask("c2c-1", lp, a, b, n1, n2 * n0)
                .task("shift-2", Fft::cyclicShiftF32, b, a, n2, n0, n1)
                .libraryTask("c2c-0", lp, a, b, n0, n1 * n2)
                .task("shift-3", Fft::cyclicShiftF32, b, a, n1, n2, n0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, a);
            TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())
                .withDevice(dev.getTornadoDevice());
            return new Plan3F32(a, b, p);
        });
    }

    private static Plan3F64 acquireF64(int n0, int n1, int n2, boolean inverse, Device dev) {
        return plan3F64Cache.computeIfAbsent(planKey(n0, n1, n2, inverse, dev), k -> {
            int total = n0 * n1 * n2;
            DoubleArray a = new DoubleArray(2 * total);
            DoubleArray b = new DoubleArray(2 * total);
            TornadoFunctions.LibraryTask4<DoubleArray, DoubleArray, Integer, Integer> lp = cufftZ2Z(inverse);
            TaskGraph g = new TaskGraph("jax4j-fft3-f64-" + k)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a)
                .libraryTask("z2z-2", lp, a, b, n2, n0 * n1)
                .task("shift-1", Fft::cyclicShiftF64, b, a, n0, n1, n2)
                .libraryTask("z2z-1", lp, a, b, n1, n2 * n0)
                .task("shift-2", Fft::cyclicShiftF64, b, a, n2, n0, n1)
                .libraryTask("z2z-0", lp, a, b, n0, n1 * n2)
                .task("shift-3", Fft::cyclicShiftF64, b, a, n1, n2, n0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, a);
            TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())
                .withDevice(dev.getTornadoDevice());
            return new Plan3F64(a, b, p);
        });
    }

    private static NDArray[] rfft3DispatchF32(NDArray x, Device dev, int n0, int n1, int n2) {
        float[] src = x.toFloatArray();
        int total = n0 * n1 * n2;
        Plan3F32 pp = acquireF32(n0, n1, n2, /*inverse=*/false, dev);
        FloatArray bufA = pp.bufA();
        for (int i = 0; i < total; i++) {
            bufA.set(2 * i, src[i]);
            bufA.set(2 * i + 1, 0f);
        }
        execPlan(pp.plan(), "cuFFT F32 3-D dispatch failed");

        int half = n2 / 2 + 1;
        float[] outRe = new float[n0 * n1 * half];
        float[] outIm = new float[n0 * n1 * half];
        for (int i = 0; i < n0; i++) {
            for (int j = 0; j < n1; j++) {
                int inRow = (i * n1 + j) * n2;
                int outRow = (i * n1 + j) * half;
                for (int k = 0; k < half; k++) {
                    outRe[outRow + k] = bufA.get(2 * (inRow + k));
                    outIm[outRow + k] = bufA.get(2 * (inRow + k) + 1);
                }
            }
        }
        Shape sh = new Shape(n0, n1, half);
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, DType.FLOAT32, dev),
            new ConcreteNDArray(outIm, sh, DType.FLOAT32, dev)
        };
    }

    private static NDArray[] rfft3DispatchF64(NDArray x, Device dev, int n0, int n1, int n2) {
        double[] src = x.toDoubleArray();
        int total = n0 * n1 * n2;
        Plan3F64 pp = acquireF64(n0, n1, n2, /*inverse=*/false, dev);
        DoubleArray bufA = pp.bufA();
        for (int i = 0; i < total; i++) {
            bufA.set(2 * i, src[i]);
            bufA.set(2 * i + 1, 0.0);
        }
        execPlan(pp.plan(), "cuFFT F64 3-D dispatch failed");

        int half = n2 / 2 + 1;
        double[] outRe = new double[n0 * n1 * half];
        double[] outIm = new double[n0 * n1 * half];
        for (int i = 0; i < n0; i++) {
            for (int j = 0; j < n1; j++) {
                int inRow = (i * n1 + j) * n2;
                int outRow = (i * n1 + j) * half;
                for (int k = 0; k < half; k++) {
                    outRe[outRow + k] = bufA.get(2 * (inRow + k));
                    outIm[outRow + k] = bufA.get(2 * (inRow + k) + 1);
                }
            }
        }
        Shape sh = new Shape(n0, n1, half);
        return new NDArray[] {
            new ConcreteNDArray(outRe, sh, dev),
            new ConcreteNDArray(outIm, sh, dev)
        };
    }

    private static NDArray irfft3DispatchF32(
            NDArray re, NDArray im, Device dev, int n0, int n1, int n2) {
        // Reconstruct the conjugate-symmetric other half on the host, then
        // do a full C2C inverse. For k in (0, n2/2), X[n2-k] = conj(X[k]);
        // across the first two axes the mirror is also index-reversed
        // (with 0 as its own mirror), matching numpy.fft.irfftn.
        int half = n2 / 2 + 1;
        float[] sRe = re.toFloatArray();
        float[] sIm = im.toFloatArray();
        int total = n0 * n1 * n2;
        Plan3F32 pp = acquireF32(n0, n1, n2, /*inverse=*/true, dev);
        FloatArray bufA = pp.bufA();
        for (int i = 0; i < n0; i++) {
            int iMir = (n0 - i) % n0;
            for (int j = 0; j < n1; j++) {
                int jMir = (n1 - j) % n1;
                for (int k = 0; k < n2; k++) {
                    boolean upper = k < half;
                    int sIdx, sign;
                    if (upper) {
                        sIdx = (i * n1 + j) * half + k;
                        sign = 1;
                    } else {
                        int kMir = n2 - k;
                        sIdx = (iMir * n1 + jMir) * half + kMir;
                        sign = -1;
                    }
                    int dIdx = 2 * ((i * n1 + j) * n2 + k);
                    bufA.set(dIdx, sRe[sIdx]);
                    bufA.set(dIdx + 1, sign * sIm[sIdx]);
                }
            }
        }
        execPlan(pp.plan(), "cuFFT F32 3-D dispatch failed");

        float[] out = new float[total];
        float scale = 1f / total;
        for (int i = 0; i < total; i++) out[i] = bufA.get(2 * i) * scale;
        return new ConcreteNDArray(out, new Shape(n0, n1, n2), DType.FLOAT32, dev);
    }

    private static NDArray irfft3DispatchF64(
            NDArray re, NDArray im, Device dev, int n0, int n1, int n2) {
        int half = n2 / 2 + 1;
        double[] sRe = re.toDoubleArray();
        double[] sIm = im.toDoubleArray();
        int total = n0 * n1 * n2;
        Plan3F64 pp = acquireF64(n0, n1, n2, /*inverse=*/true, dev);
        DoubleArray bufA = pp.bufA();
        for (int i = 0; i < n0; i++) {
            int iMir = (n0 - i) % n0;
            for (int j = 0; j < n1; j++) {
                int jMir = (n1 - j) % n1;
                for (int k = 0; k < n2; k++) {
                    boolean upper = k < half;
                    int sIdx, sign;
                    if (upper) {
                        sIdx = (i * n1 + j) * half + k;
                        sign = 1;
                    } else {
                        int kMir = n2 - k;
                        sIdx = (iMir * n1 + jMir) * half + kMir;
                        sign = -1;
                    }
                    int dIdx = 2 * ((i * n1 + j) * n2 + k);
                    bufA.set(dIdx, sRe[sIdx]);
                    bufA.set(dIdx + 1, sign * sIm[sIdx]);
                }
            }
        }
        execPlan(pp.plan(), "cuFFT F64 3-D dispatch failed");

        double[] out = new double[total];
        double scale = 1.0 / total;
        for (int i = 0; i < total; i++) out[i] = bufA.get(2 * i) * scale;
        return new ConcreteNDArray(out, new Shape(n0, n1, n2), dev);
    }

    private static boolean shapesEligibleForGpu3D(int[] d) {
        if (d.length != 3) return false;
        for (int di : d) {
            if (di < 2 || (di & (di - 1)) != 0) return false;
        }
        return true;
    }

    private static NDArray[] tryCuFft3(NDArray re, NDArray im, boolean inverse) {
        if (!cufftAvailable()) return null;
        Device dev = re.device();
        if (dev == null || dev == Device.host()) return null;
        if (!dev.equals(im.device())) return null;
        if (re.dtype() != im.dtype()) return null;
        if (!re.shape().equals(im.shape())) return null;
        if (!shapesEligibleForGpu3D(re.shape().dimensions())) return null;
        try {
            return fft3OnDevice(re, im, inverse);
        } catch (Throwable t) {
            handleDispatchFailure("fft3", t);
            return null;
        }
    }

    private static NDArray[] tryCuRfft3(NDArray x) {
        if (!cufftAvailable()) return null;
        Device dev = x.device();
        if (dev == null || dev == Device.host()) return null;
        if (!shapesEligibleForGpu3D(x.shape().dimensions())) return null;
        try {
            NDArray[] out = rfft3OnDevice(x);
            logFirstDispatchSuccess("rfft3");
            return out;
        } catch (Throwable t) {
            handleDispatchFailure("rfft3", t);
            return null;
        }
    }

    private static volatile boolean firstDispatchLogged = false;
    private static void logFirstDispatchSuccess(String kind) {
        if (firstDispatchLogged) return;
        synchronized (Fft.class) {
            if (firstDispatchLogged) return;
            firstDispatchLogged = true;
            System.err.println("[jax4j.Fft] first successful GPU dispatch: " + kind);
        }
    }

    private static NDArray tryCuIrfft3(NDArray re, NDArray im, int fullN2) {
        if (!cufftAvailable()) return null;
        Device dev = re.device();
        if (dev == null || dev == Device.host()) return null;
        if (!dev.equals(im.device())) return null;
        if (re.dtype() != im.dtype()) return null;
        if (!re.shape().equals(im.shape())) return null;
        int[] d = re.shape().dimensions();
        if (d.length != 3) return null;
        int n0 = d[0], n1 = d[1];
        if (n0 < 2 || (n0 & (n0 - 1)) != 0) return null;
        if (n1 < 2 || (n1 & (n1 - 1)) != 0) return null;
        if (fullN2 < 2 || (fullN2 & (fullN2 - 1)) != 0) return null;
        if (d[2] != fullN2 / 2 + 1) return null;
        try {
            NDArray out = irfft3OnDevice(re, im, fullN2);
            logFirstDispatchSuccess("irfft3");
            return out;
        } catch (Throwable t) {
            handleDispatchFailure("irfft3", t);
            return null;
        }
    }

    /**
     * Shared exception handler for the GPU dispatch paths. Permanently
     * disables the bridge only for errors that won't recover (class-loader
     * problems, native library missing). Transient issues — most notably
     * TornadoOutOfMemoryException, which just means the device-memory
     * pool needs to be bumped via {@code -Dtornado.device.memory=...} —
     * leave the bridge enabled so a caller who sizes the pool correctly
     * can retry.
     */
    private static void handleDispatchFailure(String kind, Throwable t) {
        String cause = t.getCause() != null ? t.getCause().getClass().getName() : "";
        boolean transientErr = cause.contains("OutOfMemory")
            || t.getClass().getName().contains("OutOfMemory");
        System.err.println("[jax4j.Fft] " + kind + "OnDevice failed → host fallback ("
            + (transientErr ? "TRANSIENT, keeping GPU path enabled" : "FATAL, disabling GPU path")
            + "): " + t.getClass().getSimpleName() + ": " + t.getMessage()
            + (t.getCause() != null ? " / cause=" + t.getCause().getMessage() : ""));
        if (!transientErr) {
            cufftAvailable = Boolean.FALSE;
        }
    }
}
