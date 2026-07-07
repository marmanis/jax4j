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

    @Override
    public String toString() {
        return name;
    }
}
