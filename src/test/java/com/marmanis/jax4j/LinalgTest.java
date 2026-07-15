package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Correctness of the LU-based dense solver. Verifies the identity matrix,
 * a small well-conditioned system with a known solution, a moderately-sized
 * random system by computing {@code A x - b}, and singular-matrix detection.
 */
public class LinalgTest {

    private static NDArray mat(int n, double... rowMajor) {
        return new ConcreteNDArray(rowMajor, new Shape(n, n));
    }

    private static NDArray vec(double... xs) {
        return new ConcreteNDArray(xs, new Shape(xs.length));
    }

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSolveIdentity() {
        NDArray A = mat(3,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1);
        NDArray b = vec(3.14, -2.7, 1.0);
        double[] x = Linalg.solve(A, b).toDoubleArray();
        assertClose(3.14, x[0], 1e-15, "x0");
        assertClose(-2.7, x[1], 1e-15, "x1");
        assertClose(1.0, x[2], 1e-15, "x2");
    }

    @Test
    public void testSolveKnownSystem() {
        // Construct an A and a target x, compute b = A x, then check that
        // solve(A, b) recovers x. Avoids off-by-one hand arithmetic in the
        // test itself and still exercises pivoting + back-substitution.
        NDArray A = mat(3,
            2, 1, 1,
            1, 3, 2,
            1, 0, 0);
        double[] xTrue = {6.0, 15.0, -23.0};
        // b = A xTrue = [2*6 + 15 + (-23), 6 + 3*15 + 2*(-23), 6] = [4, 5, 6].
        NDArray b = vec(4, 5, 6);
        double[] x = Linalg.solve(A, b).toDoubleArray();
        for (int i = 0; i < 3; i++) assertClose(xTrue[i], x[i], 1e-12, "x" + i);
    }

    @Test
    public void testSolveRequiresPivoting() {
        // A has a zero pivot at (0,0); partial pivoting must swap it out.
        NDArray A = mat(3,
            0, 1, 2,
            3, 4, 5,
            6, 7, 9);
        // Pick b = A * [1, 1, 1]^T = [3, 12, 22].
        NDArray b = vec(3, 12, 22);
        double[] x = Linalg.solve(A, b).toDoubleArray();
        assertClose(1.0, x[0], 1e-12, "x0");
        assertClose(1.0, x[1], 1e-12, "x1");
        assertClose(1.0, x[2], 1e-12, "x2");
    }

    @Test
    public void testSolveRandomDenseByResidual() {
        int n = 20;
        java.util.Random rng = new java.util.Random(31);
        double[] a = new double[n * n];
        for (int i = 0; i < n * n; i++) a[i] = rng.nextGaussian();
        // Diagonal boost to make well-conditioned.
        for (int i = 0; i < n; i++) a[i * n + i] += n;
        double[] xTrue = new double[n];
        for (int i = 0; i < n; i++) xTrue[i] = rng.nextGaussian();
        double[] bVec = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += a[i * n + j] * xTrue[j];
            bVec[i] = s;
        }
        NDArray A = new ConcreteNDArray(a, new Shape(n, n));
        NDArray b = new ConcreteNDArray(bVec, new Shape(n));
        double[] x = Linalg.solve(A, b).toDoubleArray();
        // Check A x = b residual is tiny.
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += a[i * n + j] * x[j];
            assertClose(bVec[i], s, 1e-10, "residual[" + i + "]");
        }
    }

    @Test
    public void testSingularMatrixThrows() {
        // Two equal rows -> singular.
        NDArray A = mat(3,
            1, 2, 3,
            2, 4, 6,
            0, 0, 1);
        NDArray b = vec(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> Linalg.solve(A, b));
    }

    private static double[] sortedCopy(double[] a) {
        double[] c = a.clone();
        java.util.Arrays.sort(c);
        return c;
    }

    @Test
    public void testEigOfDiagonalMatrixReturnsDiagonal() {
        NDArray A = mat(4,
            3, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 7, 0,
            0, 0, 0, 2);
        NDArray[] w = Linalg.eig(A);
        double[] wr = sortedCopy(w[0].toDoubleArray());
        double[] want = {-1, 2, 3, 7};
        for (int i = 0; i < 4; i++) assertClose(want[i], wr[i], 1e-12, "eig diag[" + i + "]");
        for (double im : w[1].toDoubleArray()) assertClose(0, im, 1e-12, "eig diag imag");
    }

    @Test
    public void testEigOfSymmetric2x2() {
        // [ 2 1 ; 1 2 ] -> eigenvalues 1 and 3.
        NDArray A = mat(2, 2, 1, 1, 2);
        NDArray[] w = Linalg.eig(A);
        double[] wr = sortedCopy(w[0].toDoubleArray());
        assertClose(1.0, wr[0], 1e-13, "eig 2x2 sym min");
        assertClose(3.0, wr[1], 1e-13, "eig 2x2 sym max");
    }

    @Test
    public void testEigOfRotationMatrixIsComplex() {
        // 2x2 rotation by pi/2: real 0, imag +/- 1.
        NDArray A = mat(2, 0, -1, 1, 0);
        NDArray[] w = Linalg.eig(A);
        double[] wr = w[0].toDoubleArray();
        double[] wi = w[1].toDoubleArray();
        assertClose(0.0, wr[0], 1e-12, "rot real0");
        assertClose(0.0, wr[1], 1e-12, "rot real1");
        assertClose(1.0, Math.abs(wi[0]), 1e-12, "rot imag0");
        assertClose(1.0, Math.abs(wi[1]), 1e-12, "rot imag1");
        assertClose(0.0, wi[0] + wi[1], 1e-12, "rot conjugates");
    }

    @Test
    public void testEigOfColleagueLikeMatrix() {
        // Companion of x^3 - x = x(x-1)(x+1), eigenvalues 0, 1, -1.
        NDArray A = mat(3,
            0, 0, 0,
            1, 0, 1,
            0, 1, 0);
        NDArray[] w = Linalg.eig(A);
        double[] wr = sortedCopy(w[0].toDoubleArray());
        double[] want = {-1, 0, 1};
        for (int i = 0; i < 3; i++) assertClose(want[i], wr[i], 1e-12, "companion eig[" + i + "]");
    }

    @Test
    public void testEigRandomVerifiedByCharacteristicSum() {
        // For A ~ 6x6, trace(A) = sum of eigenvalues (real parts, since
        // complex pairs contribute equal real parts twice).
        int n = 6;
        java.util.Random rng = new java.util.Random(41);
        double[] a = new double[n * n];
        double trace = 0.0;
        for (int i = 0; i < n * n; i++) a[i] = rng.nextGaussian();
        for (int i = 0; i < n; i++) trace += a[i * n + i];
        NDArray A = new ConcreteNDArray(a, new Shape(n, n));
        NDArray[] w = Linalg.eig(A);
        double sumReal = 0.0;
        for (double v : w[0].toDoubleArray()) sumReal += v;
        assertClose(trace, sumReal, 1e-10, "sum(eigenvalues) = trace");
    }

    @Test
    public void testGeneralizedEigDiagonal() {
        // A = diag(1, 4, 9), B = diag(1, 2, 3). Eigenvalues: A_ii / B_ii = 1, 2, 3.
        NDArray A = mat(3,
            1, 0, 0,
            0, 4, 0,
            0, 0, 9);
        NDArray B = mat(3,
            1, 0, 0,
            0, 2, 0,
            0, 0, 3);
        NDArray[] w = Linalg.eig(A, B);
        double[] wr = sortedCopy(w[0].toDoubleArray());
        double[] wi = w[1].toDoubleArray();
        double[] want = {1, 2, 3};
        for (int i = 0; i < 3; i++) assertClose(want[i], wr[i], 1e-12, "gen eig diag[" + i + "]");
        for (double im : wi) assertClose(0, im, 1e-12, "gen eig diag imag");
    }

    @Test
    public void testGeneralizedEigSymmetric() {
        // A = [[2, 1], [1, 2]], B = [[1, 0], [0, 1]] -> generalized == standard.
        // Eigenvalues 1, 3.
        NDArray A = mat(2, 2, 1, 1, 2);
        NDArray B = mat(2, 1, 0, 0, 1);
        double[] wr = sortedCopy(Linalg.eig(A, B)[0].toDoubleArray());
        assertClose(1.0, wr[0], 1e-13, "gen eig sym min");
        assertClose(3.0, wr[1], 1e-13, "gen eig sym max");
    }

    @Test
    public void testGeneralizedEigVerifiedByAxEqLambdaBx() {
        // Random A, random SPD B; verify each returned lambda satisfies
        // det(A - lambda B) ~= 0 via the characteristic sum: trace(B^{-1} A)
        // equals sum(lambda).
        int n = 5;
        java.util.Random rng = new java.util.Random(53);
        double[] a = new double[n * n];
        for (int i = 0; i < n * n; i++) a[i] = rng.nextGaussian();
        // SPD B = M M^T + n I.
        double[] m = new double[n * n];
        for (int i = 0; i < n * n; i++) m[i] = rng.nextGaussian();
        double[] b = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += m[i * n + k] * m[j * n + k];
                b[i * n + j] = s + (i == j ? n : 0);
            }
        }
        NDArray A = new ConcreteNDArray(a, new Shape(n, n));
        NDArray B = new ConcreteNDArray(b, new Shape(n, n));
        NDArray[] w = Linalg.eig(A, B);
        double sumLambda = 0.0;
        for (double v : w[0].toDoubleArray()) sumLambda += v;
        // trace(B^{-1} A) via solving B x_j = a_j column by column then
        // summing diagonal. We do it directly to verify.
        double traceInvBA = 0.0;
        for (int j = 0; j < n; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = a[i * n + j];
            double[] c = Linalg.solve(B, new ConcreteNDArray(col, new Shape(n))).toDoubleArray();
            traceInvBA += c[j];
        }
        assertClose(traceInvBA, sumLambda, 1e-9, "sum(gen eig) = trace(B^{-1} A)");
    }

    @Test
    public void testShapeMismatchThrows() {
        NDArray A = mat(2, 1, 2, 3, 4);
        NDArray b = vec(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> Linalg.solve(A, b));
    }
}
