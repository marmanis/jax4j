package com.marmanis.jax4j.ml.io;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Modules;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Java-native little-endian binary format for {@link Module} parameters.
 *
 * <p>File layout: magic {@code "JX4J"} + version byte, then for each leaf in
 * {@link Modules#flatten} order: dtype byte, rank int32, dims int32[rank],
 * raw payload bytes. Static (hyperparameter) fields are not stored — loading
 * requires a caller-supplied {@code template} module that carries them.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Serialization {
    private Serialization() {}

    private static final byte[] MAGIC = {'J', 'X', '4', 'J'};
    private static final byte VERSION = 1;

    public static void save(Module m, Path path) throws IOException {
        PyTree tree = Modules.flatten(m);
        List<NDArray> leaves = PyTrees.flatten(tree);
        try (OutputStream out = Files.newOutputStream(path)) {
            out.write(MAGIC);
            out.write(VERSION);
            for (NDArray leaf : leaves) writeLeaf(out, leaf);
        }
    }

    /**
     * Loads leaves from {@code path} into a copy of {@code template}.
     * Static fields are copied from {@code template}; only parameter arrays
     * come from disk.
     */
    public static <M extends Module> M load(M template, Path path) throws IOException {
        PyTree templateTree = Modules.flatten(template);
        List<NDArray> templateLeaves = PyTrees.flatten(templateTree);
        try (InputStream in = Files.newInputStream(path)) {
            byte[] header = in.readNBytes(4);
            if (header.length != 4 || header[0] != 'J' || header[1] != 'X' || header[2] != '4' || header[3] != 'J') {
                throw new IOException("Bad magic in " + path);
            }
            int version = in.read();
            if (version != VERSION) throw new IOException("Unsupported version: " + version);
            List<NDArray> loaded = new ArrayList<>(templateLeaves.size());
            for (int i = 0; i < templateLeaves.size(); i++) loaded.add(readLeaf(in));
            PyTree loadedTree = PyTrees.unflatten(templateTree, loaded);
            return Modules.unflatten(template, loadedTree);
        }
    }

    // ---- I/O helpers ---------------------------------------------------------

    private static void writeLeaf(OutputStream out, NDArray leaf) throws IOException {
        int[] dims = leaf.shape().dimensions();
        int headerBytes = 1 + 4 + 4 * dims.length;
        ByteBuffer header = ByteBuffer.allocate(headerBytes).order(ByteOrder.LITTLE_ENDIAN);
        header.put(dtypeByte(leaf.dtype()));
        header.putInt(dims.length);
        for (int d : dims) header.putInt(d);
        out.write(header.array());

        int size = (int) leaf.shape().size();
        int elemBytes = leaf.dtype().byteSize();
        ByteBuffer payload = ByteBuffer.allocate(size * elemBytes).order(ByteOrder.LITTLE_ENDIAN);
        switch (leaf.dtype()) {
            case FLOAT32 -> { for (float v : leaf.toFloatArray()) payload.putFloat(v); }
            case FLOAT64 -> { for (double v : leaf.toDoubleArray()) payload.putDouble(v); }
            case INT32 -> { for (int v : leaf.toIntArray()) payload.putInt(v); }
            case INT64 -> { for (long v : leaf.toLongArray()) payload.putLong(v); }
            case BOOL -> { for (boolean v : leaf.toBoolArray()) payload.put((byte) (v ? 1 : 0)); }
        }
        out.write(payload.array());
    }

    private static NDArray readLeaf(InputStream in) throws IOException {
        int dtypeByte = in.read();
        if (dtypeByte < 0) throw new IOException("Truncated file (dtype)");
        DType dtype = dtypeFromByte((byte) dtypeByte);
        int rank = readInt(in);
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) dims[i] = readInt(in);
        Shape shape = new Shape(dims);
        int size = (int) shape.size();
        int elemBytes = dtype.byteSize();
        byte[] payload = in.readNBytes(size * elemBytes);
        if (payload.length != size * elemBytes) throw new IOException("Truncated payload");
        ByteBuffer bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        return switch (dtype) {
            case FLOAT32 -> {
                float[] data = new float[size];
                for (int i = 0; i < size; i++) data[i] = bb.getFloat();
                yield new ConcreteNDArray(data, shape);
            }
            case FLOAT64 -> {
                double[] data = new double[size];
                for (int i = 0; i < size; i++) data[i] = bb.getDouble();
                yield new ConcreteNDArray(data, shape);
            }
            case INT32 -> {
                int[] data = new int[size];
                for (int i = 0; i < size; i++) data[i] = bb.getInt();
                yield new ConcreteNDArray(data, shape);
            }
            case INT64 -> {
                long[] data = new long[size];
                for (int i = 0; i < size; i++) data[i] = bb.getLong();
                yield new ConcreteNDArray(data, shape);
            }
            case BOOL -> {
                boolean[] data = new boolean[size];
                for (int i = 0; i < size; i++) data[i] = bb.get() != 0;
                yield new ConcreteNDArray(data, shape);
            }
        };
    }

    private static int readInt(InputStream in) throws IOException {
        byte[] b = in.readNBytes(4);
        if (b.length != 4) throw new IOException("Truncated file");
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    private static byte dtypeByte(DType d) {
        return switch (d) {
            case FLOAT32 -> (byte) 1;
            case FLOAT64 -> (byte) 2;
            case INT32 -> (byte) 3;
            case INT64 -> (byte) 4;
            case BOOL -> (byte) 5;
        };
    }

    private static DType dtypeFromByte(byte b) throws IOException {
        return switch (b) {
            case 1 -> DType.FLOAT32;
            case 2 -> DType.FLOAT64;
            case 3 -> DType.INT32;
            case 4 -> DType.INT64;
            case 5 -> DType.BOOL;
            default -> throw new IOException("Unknown dtype byte: " + b);
        };
    }
}
