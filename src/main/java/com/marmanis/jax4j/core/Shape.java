package com.marmanis.jax4j.core;

import java.util.Arrays;

/**
 * Represents the shape of an N-dimensional array.
 * This is an immutable record using Java 25 features.
 * 
 * @param dimensions The size of each dimension.
 */
public record Shape(int... dimensions) {
    public Shape {
        if (dimensions == null) {
            dimensions = new int[0];
        }
    }

    public int rank() {
        return dimensions.length;
    }

    public long size() {
        if (dimensions.length == 0) return 1;
        long total = 1;
        for (int dim : dimensions) {
            total *= dim;
        }
        return total;
    }

    /**
     * Computes the NumPy-style broadcast shape of two shapes, aligning
     * dimensions from the right and allowing size-1 dimensions to expand.
     *
     * @throws IllegalArgumentException if the shapes are not broadcast-compatible.
     */
    public static Shape broadcast(Shape a, Shape b) {
        int rank = Math.max(a.rank(), b.rank());
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int da = i < rank - a.rank() ? 1 : a.dimensions()[i - (rank - a.rank())];
            int db = i < rank - b.rank() ? 1 : b.dimensions()[i - (rank - b.rank())];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("Shapes " + a + " and " + b + " are not broadcast-compatible");
            }
            out[i] = Math.max(da, db);
        }
        return new Shape(out);
    }

    /** Normalizes a possibly-negative axis (e.g. {@code -1} for the last dimension) to {@code [0, rank())}. */
    public int normalizeAxis(int axis) {
        int norm = axis < 0 ? rank() + axis : axis;
        if (norm < 0 || norm >= rank()) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for shape " + this);
        }
        return norm;
    }

    /**
     * The shape resulting from reducing along {@code axis}: the axis dimension
     * is set to {@code 1} if {@code keepDims}, otherwise removed entirely
     * (rank drops by one), mirroring {@code numpy.sum(axis=, keepdims=)}.
     */
    public Shape reduceAxis(int axis, boolean keepDims) {
        int norm = normalizeAxis(axis);
        if (keepDims) {
            int[] out = dimensions.clone();
            out[norm] = 1;
            return new Shape(out);
        }
        int[] out = new int[dimensions.length - 1];
        for (int i = 0, j = 0; i < dimensions.length; i++) {
            if (i != norm) out[j++] = dimensions[i];
        }
        return new Shape(out);
    }

    /**
     * Maps a flat index in a broadcast result of the given shape back to the
     * corresponding flat index in this (possibly lower-rank / size-1-dim) shape.
     */
    public int broadcastIndex(Shape outShape, int outFlatIndex) {
        if (this.equals(outShape)) return outFlatIndex;
        int rank = outShape.rank();
        int[] outDims = outShape.dimensions();
        int[] coords = new int[rank];
        int rem = outFlatIndex;
        for (int i = rank - 1; i >= 0; i--) {
            coords[i] = rem % outDims[i];
            rem /= outDims[i];
        }
        int offset = rank - rank();
        int flat = 0;
        for (int i = 0; i < rank(); i++) {
            int dim = dimensions()[i];
            int coord = dim == 1 ? 0 : coords[i + offset];
            flat = flat * dim + coord;
        }
        return flat;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return Arrays.equals(dimensions, shape.dimensions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }

    @Override
    public String toString() {
        return Arrays.toString(dimensions);
    }
}
