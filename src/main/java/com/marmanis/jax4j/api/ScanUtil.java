package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * Leading-axis slice/stack helpers shared by {@link Lax#scan} (building the
 * per-step example shapes and the eager fallback) and {@link Grad}'s SCAN
 * forward/backward interpreters.
 */
final class ScanUtil {
    private ScanUtil() {}

    static Shape dropLeadingDim(Shape shape) {
        return new Shape(Arrays.copyOfRange(shape.dimensions(), 1, shape.rank()));
    }

    static Shape prependDim(Shape shape, int n) {
        int[] dims = new int[shape.rank() + 1];
        dims[0] = n;
        System.arraycopy(shape.dimensions(), 0, dims, 1, shape.rank());
        return new Shape(dims);
    }

    /** Extracts row {@code t} along the leading axis, dropping that axis. */
    static NDArray sliceLeading(NDArray array, int t, Shape stepShape) {
        long stepSize = stepShape.size();
        float[] src = array.toFloatArray();
        float[] dst = new float[(int) stepSize];
        System.arraycopy(src, (int) (t * stepSize), dst, 0, (int) stepSize);
        return new ConcreteNDArray(dst, stepShape);
    }

    /** Stacks equally-shaped arrays along a new leading axis, in order. */
    static NDArray stackLeading(List<NDArray> items) {
        Shape stepShape = items.get(0).shape();
        long stepSize = stepShape.size();
        float[] out = new float[(int) (stepSize * items.size())];
        for (int t = 0; t < items.size(); t++) {
            System.arraycopy(items.get(t).toFloatArray(), 0, out, (int) (t * stepSize), (int) stepSize);
        }
        return new ConcreteNDArray(out, prependDim(stepShape, items.size()));
    }
}
