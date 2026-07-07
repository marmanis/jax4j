package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class BroadcastTest {

    @Test
    public void testRowVectorPlusMatrix() {
        // (2,3) + (3,) -> (2,3), row vector broadcast across each row
        NDArray matrix = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray row = new ConcreteNDArray(new float[]{10, 20, 30}, new Shape(3));

        NDArray result = matrix.add(row);

        assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, result.toFloatArray(), 1e-6f);
        assertEquals(new Shape(2, 3), result.shape());
    }

    @Test
    public void testColumnVectorPlusMatrix() {
        // (2,3) + (2,1) -> (2,3), column vector broadcast across each column
        NDArray matrix = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));
        NDArray col = new ConcreteNDArray(new float[]{100, 200}, new Shape(2, 1));

        NDArray result = matrix.add(col);

        assertArrayEquals(new float[]{101, 102, 103, 204, 205, 206}, result.toFloatArray(), 1e-6f);
    }

    @Test
    public void testIncompatibleShapesThrow() {
        NDArray a = new ConcreteNDArray(new float[]{1, 2, 3}, new Shape(3));
        NDArray b = new ConcreteNDArray(new float[]{1, 2}, new Shape(2));
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }
}
