package com.marmanis.jax4j.sharding;

import com.marmanis.jax4j.core.Device;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An N-dimensional grid of physical devices with named axes (e.g. "data", "model").
 * Mirrors {@code jax.sharding.Mesh}.
 *
 * <p>Used in modern SPMD computing with {@link NamedSharding} and {@link ShardMap}
 * to distribute tensors and execution across accelerators.
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class DeviceMesh {

    private final Device[] devices;
    private final int[] shape;
    private final String[] axisNames;
    private final Map<String, Integer> axisNameToIndex;

    /**
     * Creates a DeviceMesh with given shape and axis names.
     *
     * @param devices   1-D array of devices ordered in row-major order of the mesh shape
     * @param shape     the dimensions of the mesh grid (e.g. [2, 4] for 8 devices)
     * @param axisNames the name of each axis (e.g. ["data", "model"])
     */
    public DeviceMesh(Device[] devices, int[] shape, String[] axisNames) {
        if (shape.length != axisNames.length) {
            throw new IllegalArgumentException("shape length (" + shape.length
                    + ") must match axisNames length (" + axisNames.length + ")");
        }
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0) throw new IllegalArgumentException("Mesh dimensions must be positive");
            total *= dim;
        }
        if (devices.length != total) {
            throw new IllegalArgumentException("devices count (" + devices.length
                    + ") does not match mesh volume (" + total + ")");
        }

        this.devices = devices.clone();
        this.shape = shape.clone();
        this.axisNames = axisNames.clone();
        this.axisNameToIndex = new HashMap<>();
        for (int i = 0; i < axisNames.length; i++) {
            if (axisNameToIndex.put(axisNames[i], i) != null) {
                throw new IllegalArgumentException("Duplicate mesh axis name: " + axisNames[i]);
            }
        }
    }

    /** Convenience constructor for a 1-D mesh. */
    public DeviceMesh(Device[] devices, String axisName) {
        this(devices, new int[]{devices.length}, new String[]{axisName});
    }

    public Device[] devices() {
        return devices.clone();
    }

    public int[] shape() {
        return shape.clone();
    }

    public String[] axisNames() {
        return axisNames.clone();
    }

    public int numDevices() {
        return devices.length;
    }

    public int axisSize(String axisName) {
        Integer idx = axisNameToIndex.get(axisName);
        if (idx == null) {
            throw new IllegalArgumentException("No such mesh axis: " + axisName);
        }
        return shape[idx];
    }

    public int axisIndex(String axisName) {
        Integer idx = axisNameToIndex.get(axisName);
        if (idx == null) {
            throw new IllegalArgumentException("No such mesh axis: " + axisName);
        }
        return idx;
    }

    /**
     * Returns the device located at the given multidimensional coordinates in the mesh.
     */
    public Device deviceAt(int... coords) {
        if (coords.length != shape.length) {
            throw new IllegalArgumentException("Expected " + shape.length + " coordinates, got " + coords.length);
        }
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            int c = coords[i];
            if (c < 0 || c >= shape[i]) {
                throw new IndexOutOfBoundsException("Coordinate " + c + " out of bounds for axis " + i + " size " + shape[i]);
            }
            flatIndex += c * stride;
            stride *= shape[i];
        }
        return devices[flatIndex];
    }
}
