package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

/**
 * SVD correctness: singular values match the diagonal of a diagonal
 * matrix and are correct for a small random 4×3 matrix (compared to a
 * reference), and A = U Σ V^T reconstruction holds to machine precision.
 */
public class LinalgSvdTest {

    private static NDArray mat(int rows, int cols, double... rowMajor) {
        return new ConcreteNDArray(rowMajor, new Shape(rows, cols));
    }

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSvdOfDiagonalMatrix() {
        NDArray A = mat(3, 3,
            3, 0, 0,
            0, 1, 0,
            0, 0, 2);
        Linalg.Svd svd = Linalg.svd(A);
        double[] sig = svd.sigma().toDoubleArray();
        // Sorted descending: 3, 2, 1.
        assertClose(3.0, sig[0], 1e-12, "sigma[0]");
        assertClose(2.0, sig[1], 1e-12, "sigma[1]");
        assertClose(1.0, sig[2], 1e-12, "sigma[2]");
    }

    @Test
    public void testSvdReconstruction() {
        // A = U Σ V^T should hold for any input.
        double[] a = new double[12];
        java.util.Random rng = new java.util.Random(29);
        for (int i = 0; i < a.length; i++) a[i] = rng.nextGaussian();
        NDArray A = mat(4, 3, a);
        Linalg.Svd svd = Linalg.svd(A);
        double[] U  = svd.U ().toDoubleArray();
        double[] s  = svd.sigma().toDoubleArray();
        double[] Vt = svd.Vt().toDoubleArray();
        // Reconstruct: R[i, j] = sum_k U[i, k] * s[k] * Vt[k, j].
        double[] R = new double[4 * 3];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0.0;
                for (int k = 0; k < 3; k++) sum += U[i * 3 + k] * s[k] * Vt[k * 3 + j];
                R[i * 3 + j] = sum;
            }
        }
        for (int i = 0; i < 12; i++) {
            assertClose(a[i], R[i], 1e-10, "recon[" + i + "]");
        }
    }

    @Test
    public void testUColumnsOrthonormal() {
        // For a well-conditioned matrix, the columns of U from svd are
        // orthonormal.
        double[] a = new double[16];
        java.util.Random rng = new java.util.Random(31);
        for (int i = 0; i < a.length; i++) a[i] = rng.nextGaussian();
        NDArray A = mat(4, 4, a);
        Linalg.Svd svd = Linalg.svd(A);
        double[] U = svd.U().toDoubleArray();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double s = 0.0;
                for (int p = 0; p < 4; p++) s += U[p * 4 + i] * U[p * 4 + j];
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, s, 1e-9, "U'U[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testVtRowsOrthonormal() {
        double[] a = new double[16];
        java.util.Random rng = new java.util.Random(37);
        for (int i = 0; i < a.length; i++) a[i] = rng.nextGaussian();
        NDArray A = mat(4, 4, a);
        Linalg.Svd svd = Linalg.svd(A);
        double[] Vt = svd.Vt().toDoubleArray();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double s = 0.0;
                for (int p = 0; p < 4; p++) s += Vt[i * 4 + p] * Vt[j * 4 + p];
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, s, 1e-11, "VtVt'[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testIllConditionedSmallSingularValues() {
        // A test that would defeat the normal-equations SVD path
        // (eig(A^T A) squares the condition number). Diagonal matrix
        // with singular values spanning 12 orders of magnitude — the
        // small ones are recoverable to full precision by
        // Golub-Kahan-Reinsch but would be lost by A^T A whose smallest
        // eigenvalue would drop below FLOAT64 precision.
        double[] targetSigmas = {1.0, 1e-3, 1e-6, 1e-9, 1e-12};
        int n = targetSigmas.length;
        double[] a = new double[n * n];
        for (int i = 0; i < n; i++) a[i * n + i] = targetSigmas[i];
        NDArray A = mat(n, n, a);
        Linalg.Svd svd = Linalg.svd(A);
        double[] s = svd.sigma().toDoubleArray();
        // Descending expected: 1, 1e-3, 1e-6, 1e-9, 1e-12.
        for (int i = 0; i < n; i++) {
            double want = targetSigmas[i];
            double got = s[i];
            double relErr = Math.abs(got - want) / want;
            if (relErr > 1e-13) {
                throw new AssertionError(String.format(
                    "sigma[%d]: expected %.3e got %.3e (rel err %.3e)",
                    i, want, got, relErr));
            }
        }
    }

    @Test
    public void testSingularValuesAreNonNegativeAndSorted() {
        double[] a = new double[15];
        java.util.Random rng = new java.util.Random(43);
        for (int i = 0; i < a.length; i++) a[i] = rng.nextGaussian();
        NDArray A = mat(5, 3, a);
        Linalg.Svd svd = Linalg.svd(A);
        double[] s = svd.sigma().toDoubleArray();
        for (int i = 0; i < s.length; i++) {
            if (s[i] < 0) throw new AssertionError("negative sigma[" + i + "] = " + s[i]);
        }
        for (int i = 0; i < s.length - 1; i++) {
            if (s[i] < s[i + 1] - 1e-12) {
                throw new AssertionError("sigma not sorted at " + i + ": " + s[i] + " < " + s[i + 1]);
            }
        }
    }
}
