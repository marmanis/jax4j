package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Numpy;
import com.marmanis.jax4j.api.Vmap;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for real INT32/BOOL dtypes: typed storage (not floats wearing a dtype
 * label), explicit {@code astype} casting, dtype-checked arithmetic, and the
 * new {@code Numpy.take} embedding-lookup primitive.
 */
public class DTypeTest {

    private static NDArray floats(float... v) { return new ConcreteNDArray(v, new Shape(v.length)); }
    private static NDArray ints(int... v) { return new ConcreteNDArray(v, new Shape(v.length)); }
    private static NDArray bools(boolean... v) { return new ConcreteNDArray(v, new Shape(v.length)); }
    private static NDArray doubles(double... v) { return new ConcreteNDArray(v, new Shape(v.length)); }
    private static NDArray longs(long... v) { return new ConcreteNDArray(v, new Shape(v.length)); }

    // ---- typed storage ----

    @Test
    void typedStorageRoundTrips() {
        assertArrayEquals(new float[]{1, 2, 3}, floats(1, 2, 3).toFloatArray(), 1e-6f);
        assertArrayEquals(new int[]{1, 2, 3}, ints(1, 2, 3).toIntArray());
        assertArrayEquals(new boolean[]{true, false}, bools(true, false).toBoolArray());
        assertEquals(DType.FLOAT32, floats(1).dtype());
        assertEquals(DType.INT32, ints(1).dtype());
        assertEquals(DType.BOOL, bools(true).dtype());
    }

    @Test
    void wrongAccessorThrows() {
        assertThrows(IllegalStateException.class, () -> ints(1, 2).toFloatArray());
        assertThrows(IllegalStateException.class, () -> bools(true).toFloatArray());
        assertThrows(IllegalStateException.class, () -> floats(1f).toIntArray());
        assertThrows(IllegalStateException.class, () -> floats(1f).toBoolArray());
    }

    @Test
    void allFiveDtypesRoundTripThroughStorage() {
        // Exercises the sealed Storage family end to end: construct, read back
        // dtype, read back the typed array. FLOAT64/INT64 are the two the
        // shorter typedStorageRoundTrips test doesn't cover.
        assertArrayEquals(new float[]{1, 2, 3}, floats(1, 2, 3).toFloatArray(), 0f);
        assertArrayEquals(new double[]{1, 2, 3}, doubles(1, 2, 3).toDoubleArray(), 0.0);
        assertArrayEquals(new int[]{1, 2, 3}, ints(1, 2, 3).toIntArray());
        assertArrayEquals(new long[]{1L, 2L, 3L}, longs(1L, 2L, 3L).toLongArray());
        assertArrayEquals(new boolean[]{true, false}, bools(true, false).toBoolArray());
        assertEquals(DType.FLOAT64, doubles(1).dtype());
        assertEquals(DType.INT64, longs(1L).dtype());
    }

    @Test
    void equalsAndHashCodeRespectDtypeAndData() {
        // Same numeric values, different dtypes -> not equal (dtype is part of
        // identity, and the storage subtype differs).
        assertNotEquals(floats(1, 2, 3), doubles(1, 2, 3));
        assertNotEquals(ints(1, 2, 3), longs(1L, 2L, 3L));
        // Same dtype + same data -> equal, with matching hashCode.
        assertEquals(doubles(1, 2, 3), doubles(1, 2, 3));
        assertEquals(doubles(1, 2, 3).hashCode(), doubles(1, 2, 3).hashCode());
        assertEquals(longs(7L, 8L), longs(7L, 8L));
        // Same dtype, different data -> not equal.
        assertNotEquals(doubles(1, 2, 3), doubles(1, 2, 4));
    }

    // ---- astype ----

    @Test
    void astypeFloatToIntTruncates() {
        NDArray x = floats(2.9f, -2.9f, 3.0f);
        assertArrayEquals(new int[]{2, -2, 3}, x.astype(DType.INT32).toIntArray());
    }

    @Test
    void astypeFloatAndIntToBoolIsNonzero() {
        assertArrayEquals(new boolean[]{false, true, true}, floats(0f, 1f, -2f).astype(DType.BOOL).toBoolArray());
        assertArrayEquals(new boolean[]{false, true}, ints(0, 5).astype(DType.BOOL).toBoolArray());
    }

    @Test
    void astypeBoolToFloatAndInt() {
        NDArray b = bools(true, false);
        assertArrayEquals(new float[]{1f, 0f}, b.astype(DType.FLOAT32).toFloatArray(), 1e-6f);
        assertArrayEquals(new int[]{1, 0}, b.astype(DType.INT32).toIntArray());
    }

    @Test
    void astypeIntToFloatWidens() {
        assertArrayEquals(new float[]{1f, 2f, 3f}, ints(1, 2, 3).astype(DType.FLOAT32).toFloatArray(), 1e-6f);
    }

    @Test
    void astypeSameDtypeIsIdentity() {
        NDArray x = floats(1, 2, 3);
        assertEquals(x, x.astype(DType.FLOAT32));
    }

    @Test
    void astypeFloat32IdentityCastStaysDifferentiable() {
        Function<NDArray, NDArray> fn = x -> x.astype(DType.FLOAT32).mul(x).sum();
        NDArray grad = JAX.grad(fn).apply(floats(3f));
        assertEquals(6f, grad.toFloatArray()[0], 1e-4f); // d/dx (x*x) = 2x
    }

    @Test
    void astypeNonFloatCastHasZeroGradient() {
        // f(x) = x.astype(BOOL).astype(FLOAT32).sum() -- non-differentiable path
        Function<NDArray, NDArray> fn = x -> x.astype(DType.BOOL).astype(DType.FLOAT32).sum();
        NDArray grad = JAX.grad(fn).apply(floats(1f, 2f, 3f));
        assertArrayEquals(new float[]{0, 0, 0}, grad.toFloatArray(), 1e-6f);
    }

    // ---- arithmetic dtype guard ----

    @Test
    void arithmeticOnNonFloatPromotes() {
        NDArray result = bools(true, false).add(floats(1f, 2f));
        assertEquals(DType.FLOAT32, result.dtype());
        assertArrayEquals(new float[]{2f, 2f}, result.toFloatArray(), 1e-6f);
    }

    @Test
    void arithmeticOnNonFloatingPromotedTypeThrows() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
            () -> ints(1).add(ints(1)));
        assertTrue(e.getMessage().contains("requires FLOAT32 or FLOAT64"));
    }

    @Test
    void arithmeticAfterExplicitCastWorks() {
        NDArray mask = bools(true, false);
        NDArray result = mask.astype(DType.FLOAT32).add(floats(1f, 1f));
        assertArrayEquals(new float[]{2f, 1f}, result.toFloatArray(), 1e-6f);
    }

    private static void assertTrue(boolean cond) {
        org.junit.jupiter.api.Assertions.assertTrue(cond);
    }

    // ---- comparisons ----

    @Test
    void comparisonDtypeMismatchPromotes() {
        NDArray result = floats(1f).gt(ints(1));
        assertEquals(DType.BOOL, result.dtype());
        assertArrayEquals(new boolean[]{false}, result.toBoolArray());
    }

    @Test
    void intComparisonsWork() {
        NDArray a = ints(1, 5, 3);
        NDArray b = ints(3, 5, 1);
        assertArrayEquals(new boolean[]{false, false, true}, a.gt(b).toBoolArray());
        assertArrayEquals(new boolean[]{false, true, false}, a.eq(b).toBoolArray());
    }

    // ---- Numpy.take (embeddings) ----

    @Test
    void takeLooksUpEmbeddingRows() {
        // table: 4 vocab x 2 dim
        NDArray table = floats(10, 11, 20, 21, 30, 31, 40, 41);
        NDArray tableShaped = new ConcreteNDArray(table.toFloatArray(), new Shape(4, 2));
        NDArray indices = ints(2, 0, 3);
        NDArray result = Numpy.take(tableShaped, indices);
        assertEquals(new Shape(3, 2), result.shape());
        assertArrayEquals(new float[]{30, 31, 10, 11, 40, 41}, result.toFloatArray(), 1e-6f);
    }

    @Test
    void takeSupportsMultiDimIndices() {
        // table: 3 vocab x 2 dim ; indices shape [2, 2] (e.g. batch x seq token ids)
        NDArray table = new ConcreteNDArray(new float[]{1, 1, 2, 2, 3, 3}, new Shape(3, 2));
        NDArray indices = new ConcreteNDArray(new int[]{0, 1, 2, 0}, new Shape(2, 2));
        NDArray result = Numpy.take(table, indices);
        assertEquals(new Shape(2, 2, 2), result.shape());
        assertArrayEquals(new float[]{1, 1, 2, 2, 3, 3, 1, 1}, result.toFloatArray(), 1e-6f);
    }

    @Test
    void takeGradientScatterAddsForRepeatedIndices() {
        // table: 3 vocab x 1 dim, all zero init via grad; indices repeat index 0 twice.
        NDArray table = floats(0f, 0f, 0f);
        NDArray tableShaped = new ConcreteNDArray(table.toFloatArray(), new Shape(3, 1));
        NDArray indices = ints(0, 0, 1);

        Function<NDArray, NDArray> fn = t -> Numpy.take(t, indices).sum();
        NDArray grad = JAX.grad(fn).apply(tableShaped);
        // index 0 looked up twice -> gradient accumulates to 2; index 1 once -> 1; index 2 unused -> 0.
        assertArrayEquals(new float[]{2f, 1f, 0f}, grad.toFloatArray(), 1e-6f);
    }

    @Test
    void takeComposesWithVmap() {
        NDArray table = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(2, 2));
        // batch of index-vectors: [[0],[1]]
        NDArray batchedIndices = new ConcreteNDArray(new int[]{0, 1}, new Shape(2, 1));
        NDArray result = Vmap.vmap(idx -> Numpy.take(table, idx)).apply(batchedIndices);
        assertEquals(new Shape(2, 1, 2), result.shape());
        assertArrayEquals(new float[]{1, 2, 3, 4}, result.toFloatArray(), 1e-6f);
    }

    // ---- FLOAT64: typed storage + full arithmetic + autodiff ----

    @Test
    void float64StorageRoundTrips() {
        NDArray x = doubles(1.5, 2.5, 3.5);
        assertEquals(DType.FLOAT64, x.dtype());
        assertArrayEquals(new double[]{1.5, 2.5, 3.5}, x.toDoubleArray(), 1e-12);
        assertThrows(IllegalStateException.class, x::toFloatArray);
    }

    @Test
    void float64ArithmeticForwardValues() {
        NDArray a = doubles(1.0, 2.0, 3.0);
        NDArray b = doubles(4.0, 5.0, 6.0);
        assertArrayEquals(new double[]{5, 7, 9}, a.add(b).toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{-3, -3, -3}, a.sub(b).toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{4, 10, 18}, a.mul(b).toDoubleArray(), 1e-12);
        assertEquals(DType.FLOAT64, a.add(b).dtype());
    }

    @Test
    void float64PreservesPrecisionBeyondFloat32() {
        // 1e16 + 1 is exactly representable in double but rounds away in float32.
        NDArray a = doubles(1.0e16);
        NDArray b = doubles(1.0);
        double result = a.add(b).toDoubleArray()[0];
        assertEquals(1.0e16 + 1.0, result, 0.0);
    }

    @Test
    void float64UnaryAndReductions() {
        NDArray x = doubles(1.0, 2.0, 3.0, 4.0);
        assertArrayEquals(new double[]{Math.exp(1), Math.exp(2), Math.exp(3), Math.exp(4)}, x.exp().toDoubleArray(), 1e-9);
        assertEquals(10.0, x.sum().toDoubleArray()[0], 1e-12);
        assertEquals(2.5, x.mean().toDoubleArray()[0], 1e-12);
    }

    @Test
    void float64DotProduct() {
        NDArray a = new ConcreteNDArray(new double[]{1, 2, 3, 4}, new Shape(2, 2));
        NDArray b = new ConcreteNDArray(new double[]{5, 6, 7, 8}, new Shape(2, 2));
        NDArray c = a.dot(b);
        assertEquals(DType.FLOAT64, c.dtype());
        assertArrayEquals(new double[]{19, 22, 43, 50}, c.toDoubleArray(), 1e-9);
    }

    @Test
    void float64GradientCorrect() {
        // d/dx x*x = 2x, computed entirely in FLOAT64.
        Function<NDArray, NDArray> fn = x -> x.mul(x).sum();
        NDArray grad = JAX.grad(fn).apply(doubles(3.0, 4.0));
        assertEquals(DType.FLOAT64, grad.dtype());
        assertArrayEquals(new double[]{6, 8}, grad.toDoubleArray(), 1e-9);
    }

    @Test
    void float64MeanAndTanhGradients() {
        Function<NDArray, NDArray> meanFn = x -> x.mean();
        NDArray meanGrad = JAX.grad(meanFn).apply(doubles(1.0, 2.0, 3.0));
        assertArrayEquals(new double[]{1.0 / 3, 1.0 / 3, 1.0 / 3}, meanGrad.toDoubleArray(), 1e-9);

        Function<NDArray, NDArray> tanhFn = x -> x.tanh().sum();
        NDArray tanhGrad = JAX.grad(tanhFn).apply(doubles(0.5));
        double expected = 1 - Math.tanh(0.5) * Math.tanh(0.5);
        assertArrayEquals(new double[]{expected}, tanhGrad.toDoubleArray(), 1e-9);
    }

    @Test
    void float32AndFloat64ArithmeticMixPromotes() {
        NDArray result = floats(1f).add(doubles(1.0));
        assertEquals(DType.FLOAT64, result.dtype());
        assertArrayEquals(new double[]{2.0}, result.toDoubleArray(), 1e-12);
    }

    @Test
    void mixedDtypeGradientMergeDoesNotThrow() {
        // Regression test for the zerosLike fix: the same FLOAT64 variable is used
        // both in a real differentiable path (sum) and a non-differentiable one
        // (gt comparison). The comparison's zero-gradient placeholder must match
        // FLOAT64, or merging it with the real FLOAT64 gradient from sum() would
        // throw a dtype-mismatch error.
        NDArray threshold = doubles(0.0);
        Function<NDArray, NDArray> fn = x -> {
            NDArray mask = x.gt(threshold).astype(DType.FLOAT64); // non-differentiable path
            NDArray ignored = mask.sum(); // not part of the returned value, just exercises the path
            return x.sum(); // real differentiable path on the same x
        };
        NDArray grad = JAX.grad(fn).apply(doubles(1.0, -1.0, 2.0));
        assertEquals(DType.FLOAT64, grad.dtype());
        assertArrayEquals(new double[]{1, 1, 1}, grad.toDoubleArray(), 1e-9);
    }

    @Test
    void astypeFloat64IdentityCastStaysDifferentiable() {
        Function<NDArray, NDArray> fn = x -> x.astype(DType.FLOAT64).mul(x).sum();
        NDArray grad = JAX.grad(fn).apply(doubles(3.0));
        assertArrayEquals(new double[]{6.0}, grad.toDoubleArray(), 1e-9);
    }

    @Test
    void astypeFloat32Float64RoundTrip() {
        NDArray f64 = doubles(1.25, 2.75);
        NDArray f32 = f64.astype(DType.FLOAT32);
        assertEquals(DType.FLOAT32, f32.dtype());
        assertArrayEquals(new float[]{1.25f, 2.75f}, f32.toFloatArray(), 1e-6f);
        assertArrayEquals(new double[]{1.25, 2.75}, f32.astype(DType.FLOAT64).toDoubleArray(), 1e-9);
    }

    // ---- INT64: typed storage + astype + comparisons ----

    @Test
    void int64StorageRoundTrips() {
        NDArray x = longs(1L, 2L, 3L);
        assertEquals(DType.INT64, x.dtype());
        assertArrayEquals(new long[]{1, 2, 3}, x.toLongArray());
        assertThrows(IllegalStateException.class, x::toIntArray);
    }

    @Test
    void int64ComparisonIsExactBeyond2Pow53() {
        // 2^53 and 2^53 + 1 are NOT both exactly representable as double -- if the
        // comparison cast through double it would (incorrectly) report them equal.
        long big = (1L << 53) + 1;
        NDArray a = longs(big);
        NDArray b = longs((1L << 53));
        assertArrayEquals(new boolean[]{true}, a.gt(b).toBoolArray());
        assertArrayEquals(new boolean[]{false}, a.eq(b).toBoolArray());
    }

    @Test
    void int64AstypeConversions() {
        NDArray x = longs(5L, -3L);
        assertArrayEquals(new float[]{5f, -3f}, x.astype(DType.FLOAT32).toFloatArray(), 1e-6f);
        assertArrayEquals(new double[]{5.0, -3.0}, x.astype(DType.FLOAT64).toDoubleArray(), 1e-9);
        assertArrayEquals(new int[]{5, -3}, x.astype(DType.INT32).toIntArray());
        assertArrayEquals(new boolean[]{true, true}, x.astype(DType.BOOL).toBoolArray());

        NDArray fromDouble = doubles(7.0).astype(DType.INT64);
        assertArrayEquals(new long[]{7L}, fromDouble.toLongArray());
        NDArray fromBool = bools(true, false).astype(DType.INT64);
        assertArrayEquals(new long[]{1L, 0L}, fromBool.toLongArray());
    }

    @Test
    void int64ArithmeticThrows() {
        assertThrows(IllegalArgumentException.class, () -> longs(1L).add(longs(2L)));
    }
}
