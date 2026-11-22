package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for Phase 2: Memory Architecture & Performance Enhancements:
 * - Panama MemorySegment backed native arrays (allocateNative and fromSegment)
 * - Zero-copy memory access via asMemorySegment()
 * - Native arithmetic and autodiff interop
 */
public class Phase2FeaturesTest {

    @Test
    public void testAllocateNativeFloat32() {
        try (Arena arena = Arena.ofConfined()) {
            Shape shape = new Shape(2, 3);
            ConcreteNDArray nativeArr = ConcreteNDArray.allocateNative(shape, DType.FLOAT32, arena);

            assertTrue(nativeArr.isNative());
            MemorySegment segment = nativeArr.asMemorySegment();
            assertNotNull(segment);
            assertEquals(6 * ValueLayout.JAVA_FLOAT.byteSize(), segment.byteSize());

            // Write through segment
            for (int i = 0; i < 6; i++) {
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (i + 1));
            }

            // Read through NDArray interface
            float[] data = nativeArr.toFloatArray();
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, data);

            // Arithmetic with heap array
            NDArray heapArr = new ConcreteNDArray(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, shape);
            NDArray sum = nativeArr.add(heapArr);

            assertArrayEquals(new float[]{11f, 22f, 33f, 44f, 55f, 66f}, sum.toFloatArray());
        }
    }

    @Test
    public void testFromSegmentFloat64() {
        try (Arena arena = Arena.ofConfined()) {
            Shape shape = new Shape(3);
            MemorySegment segment = arena.allocate(ValueLayout.JAVA_DOUBLE, 3);
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, 1.5);
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, 1, 2.5);
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, 2, 3.5);

            ConcreteNDArray arr = ConcreteNDArray.fromSegment(segment, shape, DType.FLOAT64);
            assertTrue(arr.isNative());
            assertEquals(DType.FLOAT64, arr.dtype());

            double[] data = arr.toDoubleArray();
            assertArrayEquals(new double[]{1.5, 2.5, 3.5}, data);

            // Unary op
            NDArray expArr = arr.exp();
            assertEquals(Math.exp(1.5), expArr.toDoubleArray()[0], 1e-4);
        }
    }

    @Test
    public void testHeapArrayAsMemorySegment() {
        float[] raw = new float[]{10f, 20f, 30f};
        ConcreteNDArray heapArr = new ConcreteNDArray(raw, new Shape(3));
        assertFalse(heapArr.isNative());

        MemorySegment segment = heapArr.asMemorySegment();
        assertNotNull(segment);
        assertEquals(3 * ValueLayout.JAVA_FLOAT.byteSize(), segment.byteSize());
        assertEquals(20f, segment.getAtIndex(ValueLayout.JAVA_FLOAT, 1));
    }

    @Test
    public void testAutodiffWithNativeArray() {
        try (Arena arena = Arena.ofConfined()) {
            ConcreteNDArray nativeX = ConcreteNDArray.allocateNative(new Shape(1), DType.FLOAT32, arena);
            nativeX.asMemorySegment().setAtIndex(ValueLayout.JAVA_FLOAT, 0, 4.0f);

            // f(x) = x^2 + 2x
            // df/dx = 2x + 2 = 10
            NDArray grad = Grad.grad((x) -> x.mul(x).add(x.mul(new ConcreteNDArray(new float[]{2f}, new Shape(1))))).apply(nativeX);
            assertEquals(10.0f, grad.toFloatArray()[0], 1e-4f);
        }
    }
}
