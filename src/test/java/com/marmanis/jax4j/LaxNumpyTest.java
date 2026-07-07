package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * NDArray elementwise-op and broadcasting correctness tests, modeled on the
 * role of upstream JAX's {@code tests/lax_numpy_test.py}: exercising each
 * array op against hand-computed expected values across a range of shapes.
 */
public class LaxNumpyTest {

    private static NDArray vec(float... v) {
        return new ConcreteNDArray(v, new Shape(v.length));
    }

    private static NDArray mat(int rows, int cols, float... v) {
        return new ConcreteNDArray(v, new Shape(rows, cols));
    }

    @Test
    public void testAddElementwise() {
        assertArrayEquals(new float[]{4, 6, 8}, vec(1, 2, 3).add(vec(3, 4, 5)).toFloatArray(), 1e-6f);
    }

    @Test
    public void testSubElementwise() {
        assertArrayEquals(new float[]{-2, -2, -2}, vec(1, 2, 3).sub(vec(3, 4, 5)).toFloatArray(), 1e-6f);
    }

    @Test
    public void testMulElementwise() {
        assertArrayEquals(new float[]{3, 8, 15}, vec(1, 2, 3).mul(vec(3, 4, 5)).toFloatArray(), 1e-6f);
    }

    @Test
    public void testDivElementwise() {
        assertArrayEquals(new float[]{2, 2, 2}, vec(6, 8, 10).div(vec(3, 4, 5)).toFloatArray(), 1e-6f);
    }

    @Test
    public void testScalarBroadcastAdd() {
        NDArray scalar = vec(10);
        assertArrayEquals(new float[]{11, 12, 13}, vec(1, 2, 3).add(scalar).toFloatArray(), 1e-6f);
    }

    @Test
    public void testScalarBroadcastMul() {
        NDArray scalar = vec(2);
        assertArrayEquals(new float[]{2, 4, 6}, vec(1, 2, 3).mul(scalar).toFloatArray(), 1e-6f);
    }

    @Test
    public void testRowBroadcastAcrossMatrix() {
        NDArray m = mat(2, 3, 1, 2, 3, 4, 5, 6);
        NDArray row = vec(10, 20, 30);
        assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, m.add(row).toFloatArray(), 1e-6f);
    }

    @Test
    public void testColumnBroadcastAcrossMatrix() {
        NDArray m = mat(2, 3, 1, 2, 3, 4, 5, 6);
        NDArray col = new ConcreteNDArray(new float[]{100, 200}, new Shape(2, 1));
        assertArrayEquals(new float[]{101, 102, 103, 204, 205, 206}, m.add(col).toFloatArray(), 1e-6f);
    }

    @Test
    public void testIncompatibleShapesRejected() {
        assertThrows(IllegalArgumentException.class, () -> vec(1, 2, 3).add(vec(1, 2)));
    }

    @Test
    public void testDotMatrixVectorShapes() {
        NDArray a = mat(2, 3, 1, 2, 3, 4, 5, 6);
        NDArray b = mat(3, 1, 1, 1, 1);
        NDArray result = a.dot(b);
        assertEquals(new Shape(2, 1), result.shape());
        assertArrayEquals(new float[]{6, 15}, result.toFloatArray(), 1e-6f);
    }

    @Test
    public void testDotMatrixMatrix() {
        NDArray a = mat(2, 2, 1, 2, 3, 4);
        NDArray b = mat(2, 2, 5, 6, 7, 8);
        // [[1,2],[3,4]] . [[5,6],[7,8]] = [[19,22],[43,50]]
        assertArrayEquals(new float[]{19, 22, 43, 50}, a.dot(b).toFloatArray(), 1e-6f);
    }

    @Test
    public void testSumReducesToScalar() {
        assertEquals(15.0f, vec(1, 2, 3, 4, 5).sum().toFloatArray()[0], 1e-6f);
    }

    @Test
    public void testMeanReducesToScalar() {
        assertEquals(3.0f, vec(1, 2, 3, 4, 5).mean().toFloatArray()[0], 1e-6f);
    }

    @Test
    public void testExpLogAreInverses() {
        NDArray x = vec(0.5f, 1.0f, 2.0f);
        float[] roundTrip = x.exp().log().toFloatArray();
        assertArrayEquals(x.toFloatArray(), roundTrip, 1e-4f);
    }

    @Test
    public void testSinCosPythagorean() {
        NDArray x = vec(0.3f, 1.1f, -0.7f);
        float[] sin = x.sin().toFloatArray();
        float[] cos = x.cos().toFloatArray();
        for (int i = 0; i < sin.length; i++) {
            assertEquals(1.0f, sin[i] * sin[i] + cos[i] * cos[i], 1e-4f);
        }
    }

    @Test
    public void testEqualsAndHashCodeForEqualArrays() {
        NDArray a = vec(1, 2, 3);
        NDArray b = vec(1, 2, 3);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    public void testEqualsFalseForDifferentShapes() {
        NDArray a = vec(1, 2, 3);
        NDArray b = mat(1, 3, 1, 2, 3);
        assertNotEquals(a, b);
    }

    @Test
    public void testNegativeAndFractionalValues() {
        NDArray a = vec(-1.5f, 2.5f, -3.5f);
        NDArray b = vec(0.5f, -0.5f, 1.5f);
        assertArrayEquals(new float[]{-1.0f, 2.0f, -2.0f}, a.add(b).toFloatArray(), 1e-6f);
    }
}
