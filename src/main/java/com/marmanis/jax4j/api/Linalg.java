package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * Dense real linear-algebra primitives, mirroring the shape of
 * {@code jax.numpy.linalg}. Host-only, no TornadoVM kernels, no VJP/vmap
 * rules — treated as numerical primitives alongside FFT/DCT. Both
 * {@code FLOAT32} and {@code FLOAT64} inputs are accepted; the result
 * shares the input dtype.
 *
 * <p>The initial API covers what the sibling <a
 * href="../../../../../../chebfun4j/README.md">chebfun4j</a> project needs:
 * {@link #solve} for spectral-collocation BVP solves (chebop) and
 * {@link #eig} for colleague-matrix rootfinding on Chebyshev series.
 */
public final class Linalg {
    private Linalg() {}

    /**
     * How the LU factorisation should decide that {@code A} is too close
     * to singular to solve. Passed to {@link #lu(NDArray, SingularityCheck)}
     * and {@link #solve(NDArray, NDArray, SingularityCheck)}.
     */
    public enum SingularityCheck {
        /**
         * <b>Default.</b> Growth-factor / submatrix-max check: compare each
         * pivot chosen by partial pivoting to {@code eps × ‖A‖_max}, where
         * {@code ‖A‖_max} is the largest absolute value in the original
         * matrix and {@code eps = 1e-14}. Partial pivoting picks the max of
         * the current column in the active submatrix, so this fires
         * exactly when no row in the remaining submatrix has a usable
         * entry in that column — the classical elimination-time signal of
         * rank deficiency. Elementwise-scale invariant across the whole
         * matrix (multiplying {@code A} by any nonzero constant leaves the
         * decision unchanged), but not separately invariant per row —
         * that's what {@link #SINGULARITY_CHECK_RECIPROCAL_CONDITION_EST}
         * is for. Cost: one O(n²) pass over {@code A}, then negligible per
         * elimination step.
         */
        SINGULARITY_CHECK_SUBMATRIX_MAX,

        /**
         * Reciprocal condition estimate: factor {@code A} rejecting only
         * exact-zero pivots (to avoid NaN), then estimate
         * {@code rcond = 1 / (‖A‖_∞ · ‖A⁻¹‖_∞)} via Hager's algorithm
         * (LAPACK's {@code DGECON} approach) using ~5 solves against the
         * cached factorisation. If {@code rcond < 1e-14} — i.e.,
         * {@code κ_∞(A) > 10¹⁴} — the matrix is rejected as
         * ill-conditioned. This correctly discriminates well-scaled
         * ill-conditioned matrices ({@code κ ≥ 10¹⁴}, essentially
         * numerically singular in double precision) from badly-scaled but
         * well-conditioned ones (mixed row/column scales that trip the
         * submatrix-max check without being actually near-singular). Cost:
         * a handful of extra O(n²) back-substitutions after the O(n³)
         * factorisation — small in relative terms, but not free.
         *
         * <p>For the estimate itself (not just to gate solves on it), use
         * {@link Linalg#cond(NDArray)}.
         */
        SINGULARITY_CHECK_RECIPROCAL_CONDITION_EST
    }

    /**
     * Solve the square linear system {@code A x = b} for {@code x} via
     * Gaussian elimination with partial pivoting. {@code A} must be
     * {@code n×n} and {@code b} a length-{@code n} vector; both must
     * share dtype ({@code FLOAT32} or {@code FLOAT64}). The returned
     * vector has the same dtype and length {@code n}.
     *
     * <p>Throws {@link IllegalArgumentException} if the matrix is singular
     * (all remaining pivots below {@code 1e-14 * ||A||_inf}) — no
     * least-squares fallback, no regularisation.
     */
    public static NDArray solve(NDArray A, NDArray b) {
        int n = requireSquareAndVector(A, b);
        if (A.dtype() != b.dtype()) {
            throw new IllegalArgumentException(
                "solve requires matching dtypes: " + A.dtype() + " vs " + b.dtype());
        }
        if (A.dtype() == DType.FLOAT64) {
            double[] aData = A.toDoubleArray().clone();
            double[] bData = b.toDoubleArray().clone();
            double[] x = luSolveDouble(aData, bData, n);
            return new ConcreteNDArray(x, new Shape(n));
        }
        if (A.dtype() == DType.FLOAT32) {
            // Promote to double for stability, cast back on exit — the extra
            // work is negligible next to the O(n^3) elimination and buys us
            // meaningful accuracy on ill-conditioned systems.
            double[] aData = new double[n * n];
            double[] bData = new double[n];
            float[] aSrc = A.toFloatArray();
            float[] bSrc = b.toFloatArray();
            for (int i = 0; i < n * n; i++) aData[i] = aSrc[i];
            for (int i = 0; i < n; i++) bData[i] = bSrc[i];
            double[] xD = luSolveDouble(aData, bData, n);
            float[] x = new float[n];
            for (int i = 0; i < n; i++) x[i] = (float) xD[i];
            return new ConcreteNDArray(x, new Shape(n));
        }
        throw new IllegalArgumentException("solve requires FLOAT32 or FLOAT64, got " + A.dtype());
    }

    private static int requireSquareAndVector(NDArray A, NDArray b) {
        if (A.shape().rank() != 2) {
            throw new IllegalArgumentException("solve: A must be 2-D, got shape " + A.shape());
        }
        int n = A.shape().dimensions()[0];
        if (A.shape().dimensions()[1] != n) {
            throw new IllegalArgumentException("solve: A must be square, got shape " + A.shape());
        }
        if (b.shape().rank() != 1 || b.shape().dimensions()[0] != n) {
            throw new IllegalArgumentException(
                "solve: b must be length-" + n + " vector, got shape " + b.shape());
        }
        return n;
    }

    /**
     * Generalized eigenvalues of the pair {@code (A, B)} — the values
     * {@code lambda} such that {@code A x = lambda B x} has a nonzero
     * solution {@code x}. MVP reduces to the standard eigenvalue problem
     * {@code C x = lambda x} with {@code C = B^{-1} A}, computed by
     * {@link #solve} on each column of {@code A}. Requires {@code B} to be
     * non-singular; symmetric-positive-definite {@code B} would admit a
     * Cholesky-based path that's numerically better, but the MVP takes the
     * general route.
     *
     * @return length-2 array {@code {real, imag}} of the n eigenvalues.
     */
    public static NDArray[] eig(NDArray A, NDArray B) {
        int n = requireSquareSameShape(A, B, "eig(A, B)");
        if (A.dtype() != B.dtype()) {
            throw new IllegalArgumentException(
                "eig(A, B) requires matching dtypes: " + A.dtype() + " vs " + B.dtype());
        }
        double[] Adata = (A.dtype() == DType.FLOAT64)
            ? A.toDoubleArray()
            : floatToDouble(A.toFloatArray());
        double[] Bdata = (B.dtype() == DType.FLOAT64)
            ? B.toDoubleArray().clone()
            : floatToDouble(B.toFloatArray());
        // C = B^{-1} A: for each column j of A, solve B c_j = a_j; then
        // C is the matrix whose j-th column is c_j. Bdata is modified in
        // place by luSolveDouble, so we re-factor by passing a fresh copy
        // to each solve — cost is O(n^3) per solve times n = O(n^4). Fine
        // for the small colleague-adjacent sizes chebop needs; a shared LU
        // factorization would drop this to O(n^3) total.
        double[] Cdata = new double[n * n];
        for (int j = 0; j < n; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = Adata[i * n + j];
            double[] Bcopy = Bdata.clone();
            double[] cj = luSolveDouble(Bcopy, col, n);
            for (int i = 0; i < n; i++) Cdata[i * n + j] = cj[i];
        }
        NDArray C = new ConcreteNDArray(Cdata, new Shape(n, n));
        return eig(C);
    }

    private static int requireSquareSameShape(NDArray A, NDArray B, String label) {
        if (A.shape().rank() != 2 || B.shape().rank() != 2) {
            throw new IllegalArgumentException(label + ": inputs must be 2-D");
        }
        int n = A.shape().dimensions()[0];
        if (A.shape().dimensions()[1] != n) {
            throw new IllegalArgumentException(label + ": A must be square");
        }
        if (B.shape().dimensions()[0] != n || B.shape().dimensions()[1] != n) {
            throw new IllegalArgumentException(label + ": A and B must share shape");
        }
        return n;
    }

    private static double[] floatToDouble(float[] src) {
        double[] out = new double[src.length];
        for (int i = 0; i < src.length; i++) out[i] = src[i];
        return out;
    }

    /**
     * Result of a singular-value decomposition {@code A = U diag(sigma) V^T}.
     * {@code U} is {@code m × n} with orthonormal columns,
     * {@code sigma} is a length-{@code n} vector of singular values in
     * descending order, and {@code Vt} is {@code n × n} with orthonormal
     * rows.
     */
    public record Svd(NDArray U, NDArray sigma, NDArray Vt) {}

    /**
     * Singular-value decomposition of a real {@code m × n} matrix
     * ({@code m >= n}) via the classic <b>Golub-Kahan-Reinsch</b>
     * algorithm — bidiagonalization by Householder reflectors on both
     * sides followed by an implicit-shifted QR sweep on the resulting
     * bidiagonal matrix. Numerically stable across the whole spectrum of
     * singular values (in particular, does not square the condition
     * number of {@code A} the way the normal-equations approach does);
     * accurate to close to machine precision on all singular values that
     * are above the natural noise floor of the input matrix. Returns an
     * economy SVD: {@code U} is {@code m × n}, {@code sigma} is
     * length-{@code n} in descending order, {@code Vt} is {@code n × n}.
     */
    public static Svd svd(NDArray A) {
        if (A.shape().rank() != 2) {
            throw new IllegalArgumentException("svd: A must be 2-D, got shape " + A.shape());
        }
        int m = A.shape().dimensions()[0];
        int n = A.shape().dimensions()[1];
        if (m < n) {
            throw new IllegalArgumentException(
                "svd currently requires m >= n (thin SVD), got shape " + A.shape());
        }
        double[] aData;
        if (A.dtype() == DType.FLOAT64) {
            aData = A.toDoubleArray().clone();
        } else if (A.dtype() == DType.FLOAT32) {
            aData = floatToDouble(A.toFloatArray());
        } else {
            throw new IllegalArgumentException("svd requires FLOAT32 or FLOAT64, got " + A.dtype());
        }
        // Reshape to 2-D for readability during the (already busy) algorithm.
        double[][] Aw = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) Aw[i][j] = aData[i * n + j];

        double[][] U = new double[m][n];
        double[][] V = new double[n][n];
        double[] s = new double[n];       // will hold singular values
        double[] e = new double[n];       // superdiagonal workspace

        golubKahanBidiagonalize(Aw, U, V, s, e, m, n);
        golubReinschDiagonalize(U, V, s, e, m, n);

        // Sign-canonicalize: singular values are non-negative. If any s[k]
        // came out negative from the QR sweep, absorb the sign into the
        // corresponding column of U.
        for (int k = 0; k < n; k++) {
            if (s[k] < 0.0) {
                s[k] = -s[k];
                for (int i = 0; i < m; i++) U[i][k] = -U[i][k];
            }
        }
        // Sort descending: bubble-swap columns of U, V and entries of s.
        for (int k = 0; k < n - 1; k++) {
            int max = k;
            for (int j = k + 1; j < n; j++) if (s[j] > s[max]) max = j;
            if (max != k) {
                double t = s[k]; s[k] = s[max]; s[max] = t;
                for (int i = 0; i < m; i++) { double x = U[i][k]; U[i][k] = U[i][max]; U[i][max] = x; }
                for (int i = 0; i < n; i++) { double x = V[i][k]; V[i][k] = V[i][max]; V[i][max] = x; }
            }
        }

        // Pack outputs.
        double[] Uflat = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) Uflat[i * n + j] = U[i][j];
        double[] Vt = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) Vt[i * n + j] = V[j][i];   // transpose
        return new Svd(
            new ConcreteNDArray(Uflat, new Shape(m, n)),
            new ConcreteNDArray(s,     new Shape(n)),
            new ConcreteNDArray(Vt,    new Shape(n, n))
        );
    }

    /**
     * Phase 1 of Golub-Kahan-Reinsch: reduce {@code A} to upper-
     * bidiagonal form using Householder reflectors from both sides.
     * After this routine, {@code A} has been overwritten with the
     * "compressed" Householder vectors, {@code s} holds the diagonal of
     * the bidiagonal, {@code e} holds its superdiagonal (with
     * {@code e[n-1]} left as workspace scratch), and {@code U},
     * {@code V} have been initialized to the product of the reflectors.
     */
    private static void golubKahanBidiagonalize(double[][] A, double[][] U, double[][] V,
                                                double[] s, double[] e, int m, int n) {
        int nct = Math.min(m - 1, n);           // # of column (left) reflectors
        int nrt = Math.max(0, Math.min(n - 2, m));  // # of row (right) reflectors

        for (int k = 0; k < Math.max(nct, nrt); k++) {
            if (k < nct) {
                // Householder on column k (below diagonal). Result: A[k, k] = ±||col_k||,
                // A[k+1..m-1, k] holds the (compressed) reflector vector.
                double norm = 0.0;
                for (int i = k; i < m; i++) norm = hypot(norm, A[i][k]);
                if (norm != 0.0) {
                    if (A[k][k] < 0.0) norm = -norm;
                    for (int i = k; i < m; i++) A[i][k] /= norm;
                    A[k][k] += 1.0;
                }
                s[k] = -norm;
            }
            for (int j = k + 1; j < n; j++) {
                if (k < nct && s[k] != 0.0) {
                    double t = 0.0;
                    for (int i = k; i < m; i++) t += A[i][k] * A[i][j];
                    t = -t / A[k][k];
                    for (int i = k; i < m; i++) A[i][j] += t * A[i][k];
                }
                e[j] = A[k][j];
            }
            if (k < nrt) {
                // Householder on row k (right of superdiagonal).
                double norm = 0.0;
                for (int i = k + 1; i < n; i++) norm = hypot(norm, e[i]);
                if (norm != 0.0) {
                    if (e[k + 1] < 0.0) norm = -norm;
                    for (int i = k + 1; i < n; i++) e[i] /= norm;
                    e[k + 1] += 1.0;
                }
                e[k] = -norm;
                if (k + 1 < m && e[k] != 0.0) {
                    double[] work = new double[m];
                    for (int j = k + 1; j < n; j++)
                        for (int i = k + 1; i < m; i++) work[i] += e[j] * A[i][j];
                    for (int j = k + 1; j < n; j++) {
                        double t = -e[j] / e[k + 1];
                        for (int i = k + 1; i < m; i++) A[i][j] += t * work[i];
                    }
                }
                for (int i = k + 1; i < n; i++) V[i][k] = e[i];
            }
        }

        // Set up final bidiagonal terms in s and e (handles the last row / column
        // which the loop above under-updated).
        int p = Math.min(n, m + 1);
        if (nct < n) s[nct] = A[nct][nct];
        if (m < p) s[p - 1] = 0.0;
        if (nrt + 1 < p) e[nrt] = A[nrt][p - 1];
        e[p - 1] = 0.0;

        // Generate U by applying the left reflectors (stored in the lower
        // triangle of A) to an appropriate identity.
        for (int j = nct; j < n; j++) {
            for (int i = 0; i < m; i++) U[i][j] = 0.0;
            U[j][j] = 1.0;
        }
        for (int k = nct - 1; k >= 0; k--) {
            if (s[k] != 0.0) {
                for (int j = k + 1; j < n; j++) {
                    double t = 0.0;
                    for (int i = k; i < m; i++) t += A[i][k] * U[i][j];
                    t = -t / A[k][k];
                    for (int i = k; i < m; i++) U[i][j] += t * A[i][k];
                }
                for (int i = k; i < m; i++) U[i][k] = -A[i][k];
                U[k][k] = 1.0 + U[k][k];
                for (int i = 0; i < k - 1; i++) U[i][k] = 0.0;
            } else {
                for (int i = 0; i < m; i++) U[i][k] = 0.0;
                U[k][k] = 1.0;
            }
        }

        // Generate V.
        for (int k = n - 1; k >= 0; k--) {
            if (k < nrt && e[k] != 0.0) {
                for (int j = k + 1; j < n; j++) {
                    double t = 0.0;
                    for (int i = k + 1; i < n; i++) t += V[i][k] * V[i][j];
                    t = -t / V[k + 1][k];
                    for (int i = k + 1; i < n; i++) V[i][j] += t * V[i][k];
                }
            }
            for (int i = 0; i < n; i++) V[i][k] = 0.0;
            V[k][k] = 1.0;
        }
    }

    /**
     * Phase 2 of Golub-Kahan-Reinsch: bring the bidiagonal matrix
     * (with diagonal {@code s} and superdiagonal {@code e}) to
     * diagonal form via implicit-shifted QR sweeps, accumulating the
     * Givens rotations into {@code U} (left) and {@code V} (right).
     * Handles the four "cases" of splitting/deflation the classical
     * algorithm distinguishes: negligible superdiagonal, negligible
     * diagonal, ordinary QR step, and convergence of the trailing
     * singular value.
     */
    private static void golubReinschDiagonalize(double[][] U, double[][] V, double[] s, double[] e,
                                                int m, int n) {
        int p = Math.min(n, m + 1);
        double eps = Math.pow(2.0, -52.0);
        double tiny = Math.pow(2.0, -966.0);
        int iter = 0;
        while (p > 0) {
            int k, kase;
            // Decide the current situation:
            //   kase = 1: s[p-2] is negligible and p > 0 -> deflate trailing block.
            //   kase = 2: s[k] is negligible and k < p-1  -> split at k, reduce.
            //   kase = 3: apply implicit QR sweep on [k, p-1].
            //   kase = 4: convergence — singular value at p-1 done.
            for (k = p - 2; k >= 0; k--) {
                if (k == -1) break;
                double thresh = tiny + eps * (Math.abs(s[k]) + Math.abs(s[k + 1]));
                if (Math.abs(e[k]) <= thresh || Double.isNaN(e[k])) {
                    e[k] = 0.0;
                    break;
                }
            }
            if (k == p - 2) {
                kase = 4;
            } else {
                int ks;
                for (ks = p - 1; ks >= k; ks--) {
                    if (ks == k) break;
                    double t = (ks != p     ? Math.abs(e[ks])     : 0.0)
                             + (ks != k + 1 ? Math.abs(e[ks - 1]) : 0.0);
                    if (Math.abs(s[ks]) <= tiny + eps * t) { s[ks] = 0.0; break; }
                }
                if (ks == k) kase = 3;
                else if (ks == p - 1) kase = 1;
                else { kase = 2; k = ks; }
            }
            k++;

            switch (kase) {
                case 1 -> {
                    // Deflate negligible s[p-1].
                    double f = e[p - 2];
                    e[p - 2] = 0.0;
                    for (int j = p - 2; j >= k; j--) {
                        double t = hypot(s[j], f);
                        double cs = s[j] / t;
                        double sn = f     / t;
                        s[j] = t;
                        if (j != k) {
                            f       = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        for (int i = 0; i < n; i++) {
                            double x = cs * V[i][j] + sn * V[i][p - 1];
                            V[i][p - 1] = -sn * V[i][j] + cs * V[i][p - 1];
                            V[i][j] = x;
                        }
                    }
                }
                case 2 -> {
                    // Split at negligible s[k].
                    double f = e[k - 1];
                    e[k - 1] = 0.0;
                    for (int j = k; j < p; j++) {
                        double t = hypot(s[j], f);
                        double cs = s[j] / t;
                        double sn = f     / t;
                        s[j] = t;
                        f       = -sn * e[j];
                        e[j]    = cs * e[j];
                        for (int i = 0; i < m; i++) {
                            double x = cs * U[i][j] + sn * U[i][k - 1];
                            U[i][k - 1] = -sn * U[i][j] + cs * U[i][k - 1];
                            U[i][j] = x;
                        }
                    }
                }
                case 3 -> {
                    // Implicit QR shift chase.
                    double scale = Math.max(Math.max(Math.max(Math.max(
                        Math.abs(s[p - 1]), Math.abs(s[p - 2])), Math.abs(e[p - 2])),
                        Math.abs(s[k])), Math.abs(e[k]));
                    double sp   = s[p - 1] / scale;
                    double spm1 = s[p - 2] / scale;
                    double epm1 = e[p - 2] / scale;
                    double sk   = s[k]     / scale;
                    double ek   = e[k]     / scale;
                    double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                    double c = (sp * epm1) * (sp * epm1);
                    double shift = 0.0;
                    if (b != 0.0 || c != 0.0) {
                        shift = Math.sqrt(b * b + c);
                        if (b < 0.0) shift = -shift;
                        shift = c / (b + shift);
                    }
                    double f = (sk + sp) * (sk - sp) + shift;
                    double g = sk * ek;
                    for (int j = k; j < p - 1; j++) {
                        double t  = hypot(f, g);
                        double cs = f / t;
                        double sn = g / t;
                        if (j != k) e[j - 1] = t;
                        f = cs * s[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * s[j];
                        g = sn * s[j + 1];
                        s[j + 1] = cs * s[j + 1];
                        for (int i = 0; i < n; i++) {
                            double x = cs * V[i][j] + sn * V[i][j + 1];
                            V[i][j + 1] = -sn * V[i][j] + cs * V[i][j + 1];
                            V[i][j] = x;
                        }
                        t  = hypot(f, g);
                        cs = f / t;
                        sn = g / t;
                        s[j] = t;
                        f = cs * e[j] + sn * s[j + 1];
                        s[j + 1] = -sn * e[j] + cs * s[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];
                        if (j < m - 1) {
                            for (int i = 0; i < m; i++) {
                                double x = cs * U[i][j] + sn * U[i][j + 1];
                                U[i][j + 1] = -sn * U[i][j] + cs * U[i][j + 1];
                                U[i][j] = x;
                            }
                        }
                    }
                    e[p - 2] = f;
                    iter++;
                }
                case 4 -> {
                    // Convergence at index p - 1: make s[p-1] non-negative, sort into place later.
                    if (s[k] <= 0.0) {
                        s[k] = -s[k];
                        for (int i = 0; i < n; i++) V[i][k] = -V[i][k];
                    }
                    while (k < p - 1) {
                        if (s[k] >= s[k + 1]) break;
                        double t = s[k]; s[k] = s[k + 1]; s[k + 1] = t;
                        if (k < n - 1) {
                            for (int i = 0; i < n; i++) {
                                double x = V[i][k + 1]; V[i][k + 1] = V[i][k]; V[i][k] = x;
                            }
                        }
                        if (k < m - 1) {
                            for (int i = 0; i < m; i++) {
                                double x = U[i][k + 1]; U[i][k + 1] = U[i][k]; U[i][k] = x;
                            }
                        }
                        k++;
                    }
                    iter = 0;
                    p--;
                }
                default -> throw new IllegalStateException();
            }
            if (iter > 500) throw new IllegalStateException("svd: QR sweep failed to converge");
        }
    }

    /** {@code sqrt(a² + b²)} without under/overflow. */
    private static double hypot(double a, double b) {
        return Math.hypot(a, b);
    }

    /**
     * Real eigenvalues of a general real square matrix {@code A}, returned
     * as a pair {@code {real, imag}} of length-{@code n} vectors. Complex
     * eigenvalues come out in conjugate pairs (adjacent entries with equal
     * real parts and opposite imaginary parts, following the Francis
     * quasi-triangular convention).
     *
     * <p>Implementation: Householder reduction to upper-Hessenberg form,
     * then the implicit double-shift QR algorithm with deflation — the
     * standard EISPACK/LAPACK approach. {@code FLOAT32} input is promoted
     * to {@code double} internally for stability and the result is cast
     * back on exit.
     */
    public static NDArray[] eig(NDArray A) {
        if (A.shape().rank() != 2) {
            throw new IllegalArgumentException("eig: A must be 2-D, got shape " + A.shape());
        }
        int n = A.shape().dimensions()[0];
        if (A.shape().dimensions()[1] != n) {
            throw new IllegalArgumentException("eig: A must be square, got shape " + A.shape());
        }
        double[] flat;
        if (A.dtype() == DType.FLOAT64) {
            flat = A.toDoubleArray().clone();
        } else if (A.dtype() == DType.FLOAT32) {
            float[] src = A.toFloatArray();
            flat = new double[src.length];
            for (int i = 0; i < src.length; i++) flat[i] = src[i];
        } else {
            throw new IllegalArgumentException("eig requires FLOAT32 or FLOAT64, got " + A.dtype());
        }
        double[][] H = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) H[i][j] = flat[i * n + j];
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        reduceToHessenberg(H, n);
        francisQR(H, wr, wi, n);
        if (A.dtype() == DType.FLOAT32) {
            float[] wrF = new float[n];
            float[] wiF = new float[n];
            for (int i = 0; i < n; i++) { wrF[i] = (float) wr[i]; wiF[i] = (float) wi[i]; }
            return new NDArray[]{
                new ConcreteNDArray(wrF, new Shape(n)),
                new ConcreteNDArray(wiF, new Shape(n))
            };
        }
        return new NDArray[]{
            new ConcreteNDArray(wr, new Shape(n)),
            new ConcreteNDArray(wi, new Shape(n))
        };
    }

    /**
     * Householder reduction of a general matrix to upper-Hessenberg form
     * (in place). Column {@code k} is zeroed below the subdiagonal by a
     * Householder reflector applied on both sides — eigenvalues are
     * preserved.
     */
    private static void reduceToHessenberg(double[][] H, int n) {
        for (int k = 0; k < n - 2; k++) {
            // Column norm of H[k+1..n-1, k].
            double scale = 0.0;
            for (int i = k + 1; i < n; i++) scale = Math.max(scale, Math.abs(H[i][k]));
            if (scale == 0.0) continue;
            double h = 0.0;
            double[] u = new double[n];
            for (int i = k + 1; i < n; i++) {
                u[i] = H[i][k] / scale;
                h += u[i] * u[i];
            }
            double f = u[k + 1];
            double g = (f >= 0) ? -Math.sqrt(h) : Math.sqrt(h);
            h -= f * g;
            u[k + 1] = f - g;
            // Left: H <- (I - u u^T / h) H, columns k..n-1 (rows k+1..n-1 change).
            for (int j = k; j < n; j++) {
                double s = 0.0;
                for (int i = k + 1; i < n; i++) s += u[i] * H[i][j];
                s /= h;
                for (int i = k + 1; i < n; i++) H[i][j] -= s * u[i];
            }
            // Right: H <- H (I - u u^T / h), rows 0..n-1 (columns k+1..n-1 change).
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = k + 1; j < n; j++) s += u[j] * H[i][j];
                s /= h;
                for (int j = k + 1; j < n; j++) H[i][j] -= s * u[j];
            }
            H[k + 1][k] = scale * g;
            for (int i = k + 2; i < n; i++) H[i][k] = 0.0;
        }
    }

    /**
     * Implicit double-shift QR iteration with deflation on an upper-
     * Hessenberg matrix (Francis's algorithm), extracting real eigenvalues
     * into {@code wr} and complex-conjugate imaginary parts into {@code wi}.
     * Ported from the classical EISPACK/Jama structure.
     */
    private static void francisQR(double[][] H, double[] wr, double[] wi, int nn) {
        int n = nn - 1;
        double eps = Math.pow(2.0, -52.0);
        double exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, z = 0, w, x, y;

        // Norm for convergence check.
        double norm = 0.0;
        for (int i = 0; i < nn; i++) {
            for (int j = Math.max(i - 1, 0); j < nn; j++) norm += Math.abs(H[i][j]);
        }

        int iter = 0;
        while (n >= 0) {
            // Look for small subdiagonal element to split.
            int l = n;
            while (l > 0) {
                s = Math.abs(H[l - 1][l - 1]) + Math.abs(H[l][l]);
                if (s == 0.0) s = norm;
                if (Math.abs(H[l][l - 1]) < eps * s) break;
                l--;
            }
            if (l == n) {
                // One real root found.
                H[n][n] = H[n][n] + exshift;
                wr[n] = H[n][n];
                wi[n] = 0.0;
                n--;
                iter = 0;
            } else if (l == n - 1) {
                // Two-by-two block: real pair or complex conjugate pair.
                w = H[n][n - 1] * H[n - 1][n];
                p = (H[n - 1][n - 1] - H[n][n]) / 2.0;
                q = p * p + w;
                z = Math.sqrt(Math.abs(q));
                H[n][n] += exshift;
                H[n - 1][n - 1] += exshift;
                x = H[n][n];
                if (q >= 0) {
                    // Real pair.
                    z = (p >= 0) ? p + z : p - z;
                    wr[n - 1] = x + z;
                    wr[n] = wr[n - 1];
                    if (z != 0.0) wr[n] = x - w / z;
                    wi[n - 1] = 0.0;
                    wi[n] = 0.0;
                } else {
                    // Complex conjugate pair.
                    wr[n - 1] = x + p;
                    wr[n] = x + p;
                    wi[n - 1] = z;
                    wi[n] = -z;
                }
                n -= 2;
                iter = 0;
            } else {
                // Form shift.
                x = H[n][n];
                y = 0.0;
                w = 0.0;
                if (l < n) {
                    y = H[n - 1][n - 1];
                    w = H[n][n - 1] * H[n - 1][n];
                }
                if (iter == 10 || iter == 20) {
                    // Exceptional (ad-hoc) shift, per EISPACK.
                    exshift += x;
                    for (int i = 0; i <= n; i++) H[i][i] -= x;
                    s = Math.abs(H[n][n - 1]) + Math.abs(H[n - 1][n - 2]);
                    x = 0.75 * s;
                    y = x;
                    w = -0.4375 * s * s;
                }
                iter++;
                if (iter > 300 * nn) {
                    throw new IllegalStateException(
                        "eig: Francis QR failed to converge after " + iter + " iterations");
                }
                // Look for two consecutive small subdiagonal elements.
                int m = n - 2;
                while (m >= l) {
                    z = H[m][m];
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
                    q = H[m + 1][m + 1] - z - r - s;
                    r = H[m + 2][m + 1];
                    s = Math.abs(p) + Math.abs(q) + Math.abs(r);
                    p /= s;
                    q /= s;
                    r /= s;
                    if (m == l) break;
                    if (Math.abs(H[m][m - 1]) * (Math.abs(q) + Math.abs(r))
                        < eps * Math.abs(p) * (Math.abs(H[m - 1][m - 1]) + Math.abs(z) + Math.abs(H[m + 1][m + 1]))) {
                        break;
                    }
                    m--;
                }
                for (int i = m + 2; i <= n; i++) {
                    H[i][i - 2] = 0.0;
                    if (i > m + 2) H[i][i - 3] = 0.0;
                }
                // Double QR step.
                for (int k = m; k <= n - 1; k++) {
                    boolean notLast = (k != n - 1);
                    if (k != m) {
                        p = H[k][k - 1];
                        q = H[k + 1][k - 1];
                        r = notLast ? H[k + 2][k - 1] : 0.0;
                        x = Math.abs(p) + Math.abs(q) + Math.abs(r);
                        if (x == 0.0) continue;
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                    s = Math.sqrt(p * p + q * q + r * r);
                    if (p < 0) s = -s;
                    if (s != 0.0) {
                        if (k != m) H[k][k - 1] = -s * x;
                        else if (l != m) H[k][k - 1] = -H[k][k - 1];
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        // Row modification.
                        for (int j = k; j < nn; j++) {
                            p = H[k][j] + q * H[k + 1][j];
                            if (notLast) {
                                p += r * H[k + 2][j];
                                H[k + 2][j] -= p * z;
                            }
                            H[k][j] -= p * x;
                            H[k + 1][j] -= p * y;
                        }
                        // Column modification.
                        int maxI = Math.min(n, k + 3);
                        for (int i = 0; i <= maxI; i++) {
                            p = x * H[i][k] + y * H[i][k + 1];
                            if (notLast) {
                                p += z * H[i][k + 2];
                                H[i][k + 2] -= p * r;
                            }
                            H[i][k] -= p;
                            H[i][k + 1] -= p * q;
                        }
                    }
                }
            }
        }
    }

    private static double[] luSolveDouble(double[] a, double[] b, int n) {
        // Precompute a rough scale for singularity check.
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) rowSum += Math.abs(a[i * n + j]);
            if (rowSum > norm) norm = rowSum;
        }
        double singTol = 1e-14 * Math.max(norm, 1.0);
        // Forward elimination with partial pivoting.
        for (int k = 0; k < n; k++) {
            int piv = k;
            double pivAbs = Math.abs(a[k * n + k]);
            for (int i = k + 1; i < n; i++) {
                double v = Math.abs(a[i * n + k]);
                if (v > pivAbs) { pivAbs = v; piv = i; }
            }
            if (pivAbs <= singTol) {
                throw new IllegalArgumentException(
                    "solve: matrix is singular (pivot " + pivAbs + " below tolerance " + singTol + ")");
            }
            if (piv != k) {
                for (int j = 0; j < n; j++) {
                    double t = a[k * n + j];
                    a[k * n + j] = a[piv * n + j];
                    a[piv * n + j] = t;
                }
                double t = b[k]; b[k] = b[piv]; b[piv] = t;
            }
            double pivInv = 1.0 / a[k * n + k];
            for (int i = k + 1; i < n; i++) {
                double factor = a[i * n + k] * pivInv;
                if (factor == 0.0) continue;
                a[i * n + k] = 0.0;
                for (int j = k + 1; j < n; j++) {
                    a[i * n + j] -= factor * a[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }
        // Back-substitute.
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int j = i + 1; j < n; j++) s -= a[i * n + j] * x[j];
            x[i] = s / a[i * n + i];
        }
        return x;
    }
}
