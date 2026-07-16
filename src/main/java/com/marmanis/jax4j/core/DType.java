package com.marmanis.jax4j.core;

/**
 * Supported data types in jax4j.
 */
public enum DType {
    FLOAT32(Float.BYTES, "float32"),
    FLOAT64(Double.BYTES, "float64"),
    INT32(Integer.BYTES, "int32"),
    INT64(Long.BYTES, "int64"),
    BOOL(1, "bool");

    private final int byteSize;
    private final String name;

    DType(int byteSize, String name) {
        this.byteSize = byteSize;
        this.name = name;
    }

    public int byteSize() {
        return byteSize;
    }

    public static DType promote(DType d1, DType d2) {
        if (d1 == d2) return d1;
        if (d1 == BOOL) return d2;
        if (d2 == BOOL) return d1;

        if (d1 == FLOAT64 || d2 == FLOAT64) return FLOAT64;

        if ((d1 == INT64 && d2 == FLOAT32) || (d1 == FLOAT32 && d2 == INT64)) {
            return FLOAT64;
        }

        if (d1 == FLOAT32 || d2 == FLOAT32) return FLOAT32;
        if (d1 == INT64 || d2 == INT64) return INT64;

        return INT32;
    }

    @Override
    public String toString() {
        return name;
    }
}
