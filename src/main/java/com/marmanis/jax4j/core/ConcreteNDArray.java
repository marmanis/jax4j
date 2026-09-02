package com.marmanis.jax4j.core;

import com.marmanis.jax4j.backend.ExecutionBackend;
import com.marmanis.jax4j.backend.HostBackend;
import com.marmanis.jax4j.backend.TornadoVMBackend;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.tracing.Tracer;
import com.marmanis.jax4j.tracing.TracedNDArray;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * A concrete implementation of NDArray that holds actual data in host memory.
 * Storage is genuinely typed and held in a single {@link Storage} value — a
 * sealed family with exactly one record per dtype, each wrapping one primitive
 * array. There is no "float standing in for bool/int" representation anywhere,
 * and — unlike the earlier five-parallel-nullable-fields layout — no way to
 * represent a dtype/data mismatch: the record subtype <em>is</em> the dtype.
 * Internal dispatch is a compiler-checked exhaustive {@code switch} over the
 * storage rather than a hand-maintained {@code if (dtype == ...)} chain.
 *
 * <p>Arithmetic (`add`/`mul`/.../`dot`/reductions) requires both operands to
 * share the same <em>floating</em> dtype (FLOAT32 or FLOAT64) and throws
 * otherwise; {@link #astype} is the explicit, principled way to convert.
 * FLOAT64 arithmetic always runs on the host (no TornadoVM kernel), unlike
 * FLOAT32 which dispatches to a device when explicitly placed.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class ConcreteNDArray implements NDArray {
    // Threshold above which elementwise and unary ops use parallel streams.
    private static final int PARALLEL_THRESHOLD = 65_536;
    private final Shape shape;
    private final DType dtype;
    private final Device device;
    private final Storage storage;

    /**
     * Typed backing store for a {@link ConcreteNDArray}: a sealed family with
     * one record per dtype. Holding storage as a single sealed value (rather
     * than five parallel nullable arrays) makes the dtype and the data
     * inseparable, and lets every dtype-dependent method dispatch via an
     * exhaustive {@code switch} the compiler checks. Array equality/hashing/
     * printing lives here so callers don't reach back into raw arrays.
     */
    private sealed interface Storage {
        DType dtype();
        boolean dataEquals(Storage other);
        int dataHashCode();
        String dataString();
    }

    private record F32Storage(float[] data) implements Storage {
        @Override public DType dtype() { return DType.FLOAT32; }
        @Override public boolean dataEquals(Storage o) { return o instanceof F32Storage f && Arrays.equals(data, f.data); }
        @Override public int dataHashCode() { return Arrays.hashCode(data); }
        @Override public String dataString() { return Arrays.toString(data); }
    }

    private record F64Storage(double[] data) implements Storage {
        @Override public DType dtype() { return DType.FLOAT64; }
        @Override public boolean dataEquals(Storage o) { return o instanceof F64Storage f && Arrays.equals(data, f.data); }
        @Override public int dataHashCode() { return Arrays.hashCode(data); }
        @Override public String dataString() { return Arrays.toString(data); }
    }

    private record I32Storage(int[] data) implements Storage {
        @Override public DType dtype() { return DType.INT32; }
        @Override public boolean dataEquals(Storage o) { return o instanceof I32Storage f && Arrays.equals(data, f.data); }
        @Override public int dataHashCode() { return Arrays.hashCode(data); }
        @Override public String dataString() { return Arrays.toString(data); }
    }

    private record I64Storage(long[] data) implements Storage {
        @Override public DType dtype() { return DType.INT64; }
        @Override public boolean dataEquals(Storage o) { return o instanceof I64Storage f && Arrays.equals(data, f.data); }
        @Override public int dataHashCode() { return Arrays.hashCode(data); }
        @Override public String dataString() { return Arrays.toString(data); }
    }

    private record BoolStorage(boolean[] data) implements Storage {
        @Override public DType dtype() { return DType.BOOL; }
        @Override public boolean dataEquals(Storage o) { return o instanceof BoolStorage f && Arrays.equals(data, f.data); }
        @Override public int dataHashCode() { return Arrays.hashCode(data); }
        @Override public String dataString() { return Arrays.toString(data); }
    }

    /** Canonical constructor. Every public constructor funnels through here. */
    private ConcreteNDArray(Storage storage, Shape shape, Device device) {
        this.storage = storage;
        this.shape = shape;
        this.dtype = storage.dtype();
        this.device = device;
    }

    public ConcreteNDArray(float[] data, Shape shape, DType dtype, Device device) {
        this(f32Storage(data, dtype), shape, device);
    }

    private static F32Storage f32Storage(float[] data, DType dtype) {
        if (dtype != DType.FLOAT32) {
            throw new IllegalArgumentException("This constructor only accepts dtype FLOAT32, got " + dtype);
        }
        return new F32Storage(data);
    }

    public ConcreteNDArray(float[] data, Shape shape) {
        this(new F32Storage(data), shape, Device.defaultDevice());
    }

    public ConcreteNDArray(int[] data, Shape shape, Device device) {
        this(new I32Storage(data), shape, device);
    }

    public ConcreteNDArray(int[] data, Shape shape) {
        this(new I32Storage(data), shape, Device.defaultDevice());
    }

    public ConcreteNDArray(boolean[] data, Shape shape, Device device) {
        this(new BoolStorage(data), shape, device);
    }

    public ConcreteNDArray(boolean[] data, Shape shape) {
        this(new BoolStorage(data), shape, Device.defaultDevice());
    }

    public ConcreteNDArray(double[] data, Shape shape, Device device) {
        this(new F64Storage(data), shape, device);
    }

    public ConcreteNDArray(double[] data, Shape shape) {
        this(new F64Storage(data), shape, Device.defaultDevice());
    }

    public ConcreteNDArray(long[] data, Shape shape, Device device) {
        this(new I64Storage(data), shape, device);
    }

    public ConcreteNDArray(long[] data, Shape shape) {
        this(new I64Storage(data), shape, Device.defaultDevice());
    }

    @Override public Shape shape() { return shape; }
    @Override public DType dtype() { return dtype; }
    @Override public Device device() { return device; }

    @Override
    public NDArray to(Device targetDevice) {
        return new ConcreteNDArray(storage, shape, targetDevice);
    }

    // Typed views of the backing store. Each is only called from a branch
    // that has already established the matching dtype (via the requireFloating
    // guards, an explicit dtype check, or a storage switch), so the cast is
    // always safe.
    private float[]   f32() { return ((F32Storage) storage).data(); }
    private double[]  f64() { return ((F64Storage) storage).data(); }
    private int[]     i32() { return ((I32Storage) storage).data(); }
    private long[]    i64() { return ((I64Storage) storage).data(); }
    private boolean[] bl()  { return ((BoolStorage) storage).data(); }

    private void requireDtype(DType expected, String accessor) {
        if (dtype != expected) {
            throw new IllegalStateException(accessor + "() requires dtype " + expected + ", but this array is " + dtype);
        }
    }

    /** True for dtypes that support arithmetic ({@link #elementwise}/{@link #unary}/{@link #dot}/reductions). */
    private static boolean isFloating(DType d) {
        return d == DType.FLOAT32 || d == DType.FLOAT64;
    }

    private void requireFloatingDtype(String opName) {
        if (!isFloating(dtype)) {
            throw new IllegalArgumentException(
                opName + " requires FLOAT32 or FLOAT64 operands, got " + dtype + " — use .astype() to convert.");
        }
    }

    private void requireSameFloatingDtype(NDArray other, String opName) {
        requireFloatingDtype(opName);
        if (other.dtype() != dtype) {
            throw new IllegalArgumentException(
                opName + " requires both operands to share the same floating dtype, got " + dtype
                    + " and " + other.dtype() + " — use .astype() to convert.");
        }
    }

    private static ExecutionBackend backendFor(Device d) {
        return d.getTornadoDevice() != null ? TornadoVMBackend.INSTANCE : HostBackend.INSTANCE;
    }

    private boolean isTracing() {
        return Tracer.current() != null;
    }

    private NDArray toTraced() {
        Tracer tracer = Tracer.current();
        return new TracedNDArray(tracer.nextConstant(this));
    }

    /**
     * Applies a binary elementwise op with NumPy-style broadcasting. Requires
     * both operands to share the same floating dtype (FLOAT32 or FLOAT64).
     *
     * <p>Device dispatch (FLOAT32 only): if both operands sit on {@link
     * Device#host()}, this runs the same plain Java loop as always (zero
     * behavior change). If either operand was explicitly placed on a non-host
     * device via {@link #to}, both operands must be on the <em>same</em>
     * device — mismatches throw, mirroring JAX's refusal to silently move data
     * across devices — and, when the shapes match exactly (no broadcasting
     * needed), the op dispatches to that device's {@link ExecutionBackend}
     * instead of running on the host. Broadcasting itself is not yet supported
     * by the device backends, so a shape mismatch on a non-host device still
     * computes on the host but keeps the device tag. FLOAT64 always runs the
     * host loop — {@link ExecutionBackend} is {@code float[]}-only.
     */
    private NDArray elementwise(NDArray other, Primitive primitive, DoubleBinaryOperator op,
                                 java.util.function.BiFunction<NDArray, NDArray, NDArray> tracedOp) {
        if (dtype != other.dtype() || other instanceof TracedNDArray || isTracing()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            if (left instanceof TracedNDArray || right instanceof TracedNDArray || isTracing()) {
                NDArray leftTraced = left instanceof TracedNDArray ? left : ((ConcreteNDArray) left).toTraced();
                NDArray rightTraced = right instanceof TracedNDArray ? right : ((ConcreteNDArray) right).toTraced();
                return tracedOp.apply(leftTraced, rightTraced);
            }
            return ((ConcreteNDArray) left).elementwise(right, primitive, op, tracedOp);
        }

        requireSameFloatingDtype(other, primitive.toString());

        if (dtype == DType.FLOAT64) {
            double[] thisData = f64();
            double[] otherData = other.toDoubleArray();
            Shape outShape = Shape.broadcast(shape, other.shape());
            double[] result = new double[(int) outShape.size()];
            Shape thisShape = shape;
            Shape otherShape = other.shape();
            if (result.length > PARALLEL_THRESHOLD) {
                IntStream.range(0, result.length).parallel().forEach(i -> {
                    double v1 = thisData[thisShape.broadcastIndex(outShape, i)];
                    double v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                    result[i] = op.applyAsDouble(v1, v2);
                });
            } else {
                for (int i = 0; i < result.length; i++) {
                    double v1 = thisData[thisShape.broadcastIndex(outShape, i)];
                    double v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                    result[i] = op.applyAsDouble(v1, v2);
                }
            }
            return new ConcreteNDArray(result, outShape, device);
        }

        float[] thisData = f32();
        float[] otherData = other.toFloatArray();
        Device otherDevice = other.device();
        boolean hostOnly = device.equals(Device.host()) && otherDevice.equals(Device.host());

        if (!hostOnly) {
            if (!device.equals(otherDevice)) {
                throw new IllegalStateException(
                    "Cannot combine arrays on different devices: " + device + " vs " + otherDevice);
            }
            if (shape.equals(other.shape())) {
                float[] result = backendFor(device).binary(primitive, thisData, otherData, device);
                return new ConcreteNDArray(result, shape, dtype, device);
            }
        }

        Shape outShape = Shape.broadcast(shape, other.shape());
        float[] result = new float[(int) outShape.size()];
        Shape thisShape = shape;
        Shape otherShape = other.shape();

        if (result.length > PARALLEL_THRESHOLD) {
            IntStream.range(0, result.length).parallel().forEach(i -> {
                float v1 = thisData[thisShape.broadcastIndex(outShape, i)];
                float v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                result[i] = (float) op.applyAsDouble(v1, v2);
            });
        } else {
            for (int i = 0; i < result.length; i++) {
                float v1 = thisData[thisShape.broadcastIndex(outShape, i)];
                float v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                result[i] = (float) op.applyAsDouble(v1, v2);
            }
        }
        return new ConcreteNDArray(result, outShape, dtype, device);
    }

    @Override
    public NDArray add(NDArray other) {
        return elementwise(other, Primitive.ADD, (a, b) -> a + b, NDArray::add);
    }

    @Override
    public NDArray sub(NDArray other) {
        return elementwise(other, Primitive.SUB, (a, b) -> a - b, NDArray::sub);
    }

    @Override
    public NDArray mul(NDArray other) {
        return elementwise(other, Primitive.MUL, (a, b) -> a * b, NDArray::mul);
    }

    @Override
    public NDArray div(NDArray other) {
        return elementwise(other, Primitive.DIV, (a, b) -> a / b, NDArray::div);
    }

    @Override public NDArray max(NDArray other) { return elementwise(other, Primitive.MAX, Math::max, NDArray::max); }
    @Override public NDArray min(NDArray other) { return elementwise(other, Primitive.MIN, Math::min, NDArray::min); }

    /**
     * Applies a broadcasting comparison, producing a real {@code DType.BOOL}
     * result. Both operands must share a dtype (FLOAT32, INT32, FLOAT64, or
     * INT64); mismatches throw rather than silently coercing one side.
     * FLOAT32/INT32/FLOAT64 compare via {@code test} (a lossless double
     * comparison in every one of those cases); INT64 compares as a genuine
     * {@code long} by switching on {@code primitive} directly, since casting
     * a long through double would lose precision beyond 2^53. BOOL is not
     * comparable and throws.
     */
    private NDArray compareElementwise(NDArray other, Primitive primitive,
                                        java.util.function.BiPredicate<Double, Double> test,
                                        java.util.function.BiFunction<NDArray, NDArray, NDArray> tracedOp) {
        if (dtype != other.dtype() || other instanceof TracedNDArray || isTracing()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            if (left instanceof TracedNDArray || right instanceof TracedNDArray || isTracing()) {
                NDArray leftTraced = left instanceof TracedNDArray ? left : ((ConcreteNDArray) left).toTraced();
                NDArray rightTraced = right instanceof TracedNDArray ? right : ((ConcreteNDArray) right).toTraced();
                return tracedOp.apply(leftTraced, rightTraced);
            }
            return ((ConcreteNDArray) left).compareElementwise(right, primitive, test, tracedOp);
        }

        Shape outShape = Shape.broadcast(shape, other.shape());
        boolean[] result = new boolean[(int) outShape.size()];

        switch (storage) {
            case F32Storage s -> {
                float[] thisData = s.data();
                float[] otherData = other.toFloatArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = thisData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case F64Storage s -> {
                double[] thisData = s.data();
                double[] otherData = other.toDoubleArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = thisData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case I32Storage s -> {
                // Compared as double (lossless for int32, unlike a float cast) to
                // avoid precision loss for indices beyond float32's 24-bit mantissa.
                int[] thisData = s.data();
                int[] otherData = other.toIntArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = thisData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case I64Storage s -> {
                long[] thisData = s.data();
                long[] otherData = other.toLongArray();
                for (int i = 0; i < result.length; i++) {
                    long v1 = thisData[shape.broadcastIndex(outShape, i)];
                    long v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = switch (primitive) {
                        case GT -> v1 > v2;
                        case GE -> v1 >= v2;
                        case LT -> v1 < v2;
                        case LE -> v1 <= v2;
                        case EQ -> v1 == v2;
                        case NE -> v1 != v2;
                        default -> throw new IllegalStateException("Not a comparison primitive: " + primitive);
                    };
                }
            }
            case BoolStorage s -> throw new IllegalArgumentException(
                primitive + " requires FLOAT32, INT32, FLOAT64, or INT64 operands, got " + dtype);
        }
        return new ConcreteNDArray(result, outShape, device);
    }

    @Override public NDArray gt(NDArray other) { return compareElementwise(other, Primitive.GT, (a, b) -> a > b, NDArray::gt); }
    @Override public NDArray ge(NDArray other) { return compareElementwise(other, Primitive.GE, (a, b) -> a >= b, NDArray::ge); }
    @Override public NDArray lt(NDArray other) { return compareElementwise(other, Primitive.LT, (a, b) -> a < b, NDArray::lt); }
    @Override public NDArray le(NDArray other) { return compareElementwise(other, Primitive.LE, (a, b) -> a <= b, NDArray::le); }
    @Override public NDArray eq(NDArray other) { return compareElementwise(other, Primitive.EQ, (a, b) -> a.doubleValue() == b.doubleValue(), NDArray::eq); }
    @Override public NDArray ne(NDArray other) { return compareElementwise(other, Primitive.NE, (a, b) -> a.doubleValue() != b.doubleValue(), NDArray::ne); }

    @Override
    public NDArray dot(NDArray other) {
        if (dtype != other.dtype() || other instanceof TracedNDArray || isTracing()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            if (left instanceof TracedNDArray || right instanceof TracedNDArray || isTracing()) {
                NDArray leftTraced = left instanceof TracedNDArray ? left : ((ConcreteNDArray) left).toTraced();
                NDArray rightTraced = right instanceof TracedNDArray ? right : ((ConcreteNDArray) right).toTraced();
                return leftTraced.dot(rightTraced);
            }
            return ((ConcreteNDArray) left).dot(right);
        }
        // Batched matmul: any rank > 2 goes through MATMUL primitive path.
        if (shape.rank() > 2 || other.shape().rank() > 2) {
            requireSameFloatingDtype(other, "matmul");
            return matmulEager(this, other);
        }
        requireSameFloatingDtype(other, "dot");
        int M = shape.dimensions()[0];
        int K = shape.dimensions()[1];
        int N = other.shape().dimensions()[1];

        if (dtype == DType.FLOAT64) {
            double[] thisData = f64();
            double[] otherData = other.toDoubleArray();
            double[] result = new double[M * N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0;
                    for (int k = 0; k < K; k++) sum += thisData[i * K + k] * otherData[k * N + j];
                    result[i * N + j] = sum;
                }
            }
            return new ConcreteNDArray(result, new Shape(M, N), device);
        }

        float[] thisData = f32();
        float[] otherData = other.toFloatArray();
        Device otherDevice = other.device();

        boolean hostOnly = device.equals(Device.host()) && otherDevice.equals(Device.host());
        if (!hostOnly) {
            if (!device.equals(otherDevice)) {
                throw new IllegalStateException(
                    "Cannot combine arrays on different devices: " + device + " vs " + otherDevice);
            }
            float[] result = backendFor(device).matmul(thisData, otherData, M, K, N, device);
            return new ConcreteNDArray(result, new Shape(M, N), dtype, device);
        }

        float[] result = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += thisData[i * K + k] * otherData[k * N + j];
                result[i * N + j] = sum;
            }
        }
        return new ConcreteNDArray(result, new Shape(M, N), dtype, device);
    }

    @Override public NDArray sum() {
        if (isTracing()) return toTraced().sum();
        requireFloatingDtype("sum");
        ExecutionBackend backend = backendFor(device);
        if (dtype == DType.FLOAT64) {
            double[] res = backend.reduce(Primitive.SUM, f64(), device);
            return new ConcreteNDArray(res, new Shape(1), device);
        }
        float[] res = backend.reduce(Primitive.SUM, f32(), device);
        return new ConcreteNDArray(res, new Shape(1), dtype, device);
    }

    @Override
    public NDArray mean() {
        if (isTracing()) return toTraced().mean();
        requireFloatingDtype("mean");
        ExecutionBackend backend = backendFor(device);
        if (dtype == DType.FLOAT64) {
            double[] res = backend.reduce(Primitive.MEAN, f64(), device);
            return new ConcreteNDArray(res, new Shape(1), device);
        }
        float[] res = backend.reduce(Primitive.MEAN, f32(), device);
        return new ConcreteNDArray(res, new Shape(1), dtype, device);
    }

    @Override
    public NDArray sum(int axis, boolean keepDims) {
        if (isTracing()) return toTraced().sum(axis, keepDims);
        requireFloatingDtype("sum");
        return reduceAxis(axis, keepDims, false);
    }

    @Override
    public NDArray mean(int axis, boolean keepDims) {
        if (isTracing()) return toTraced().mean(axis, keepDims);
        requireFloatingDtype("mean");
        return reduceAxis(axis, keepDims, true);
    }

    /**
     * Reduces along a single axis: contiguous row-major data decomposes into
     * {@code outerSize * axisSize * innerSize} blocks, so every (outer, inner)
     * pair sums the {@code axisSize} elements at stride {@code innerSize}.
     * Never dispatches to a device (mirrors {@link #sum()}/{@link #mean()},
     * which are always host-computed — TornadoVM reduction kernels are a
     * follow-up, same as for the full-reduction primitives).
     */
    private NDArray reduceAxis(int axis, boolean keepDims, boolean mean) {
        int norm = shape.normalizeAxis(axis);
        int[] dims = shape.dimensions();
        int axisSize = dims[norm];
        int outerSize = 1;
        for (int i = 0; i < norm; i++) outerSize *= dims[i];
        int innerSize = 1;
        for (int i = norm + 1; i < dims.length; i++) innerSize *= dims[i];

        Shape outShape = shape.reduceAxis(norm, keepDims);

        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            double[] out = backendFor(device).reduceAxis(mean ? Primitive.MEAN : Primitive.SUM, in, outerSize, axisSize, innerSize, device);
            return new ConcreteNDArray(out, outShape, device);
        }

        float[] in = f32();
        float[] out = backendFor(device).reduceAxis(mean ? Primitive.MEAN : Primitive.SUM, in, outerSize, axisSize, innerSize, device);
        return new ConcreteNDArray(out, outShape, dtype, device);
    }

    @Override
    public NDArray max(int axis, boolean keepDims) {
        if (isTracing()) return toTraced().max(axis, keepDims);
        requireFloatingDtype("max");
        return maxMinAxis(axis, keepDims, true);
    }

    @Override
    public NDArray min(int axis, boolean keepDims) {
        if (isTracing()) return toTraced().min(axis, keepDims);
        requireFloatingDtype("min");
        return maxMinAxis(axis, keepDims, false);
    }

    private NDArray maxMinAxis(int axis, boolean keepDims, boolean isMax) {
        int norm = shape.normalizeAxis(axis);
        int[] dims = shape.dimensions();
        int axisSize = dims[norm];
        int outerSize = 1;
        for (int i = 0; i < norm; i++) outerSize *= dims[i];
        int innerSize = 1;
        for (int i = norm + 1; i < dims.length; i++) innerSize *= dims[i];
        Shape outShape = shape.reduceAxis(norm, keepDims);

        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            double[] out = new double[(int) outShape.size()];
            for (int o = 0; o < outerSize; o++) {
                for (int inr = 0; inr < innerSize; inr++) {
                    double best = in[o * axisSize * innerSize + inr];
                    for (int a = 1; a < axisSize; a++) {
                        double v = in[o * axisSize * innerSize + a * innerSize + inr];
                        if (isMax ? v > best : v < best) best = v;
                    }
                    out[o * innerSize + inr] = best;
                }
            }
            return new ConcreteNDArray(out, outShape, device);
        }
        float[] in = f32();
        float[] out = new float[(int) outShape.size()];
        for (int o = 0; o < outerSize; o++) {
            for (int inr = 0; inr < innerSize; inr++) {
                float best = in[o * axisSize * innerSize + inr];
                for (int a = 1; a < axisSize; a++) {
                    float v = in[o * axisSize * innerSize + a * innerSize + inr];
                    if (isMax ? v > best : v < best) best = v;
                }
                out[o * innerSize + inr] = best;
            }
        }
        return new ConcreteNDArray(out, outShape, dtype, device);
    }

    @Override
    public NDArray slice(int[] starts, int[] stops, int[] steps) {
        if (isTracing()) return toTraced().slice(starts, stops, steps);
        int rank = shape.rank();
        if (starts.length != rank || stops.length != rank || steps.length != rank) {
            throw new IllegalArgumentException(
                "slice: starts/stops/steps must all have length " + rank + ", got "
                    + starts.length + "/" + stops.length + "/" + steps.length);
        }
        int[] inDims = shape.dimensions();
        int[] outDims = new int[rank];
        int[] realStops = new int[rank];
        for (int i = 0; i < rank; i++) {
            if (steps[i] <= 0) throw new IllegalArgumentException("slice: step must be positive, got " + steps[i]);
            int stop = stops[i] == -1 ? inDims[i] : stops[i];
            realStops[i] = stop;
            int span = stop - starts[i];
            outDims[i] = span <= 0 ? 0 : (span + steps[i] - 1) / steps[i];
        }
        Shape outShape = new Shape(outDims);
        int size = (int) outShape.size();

        int[] inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inDims[i + 1];

        switch (storage) {
            case F32Storage s -> {
                float[] in = s.data();
                float[] out = new float[size];
                sliceCopyF32(in, out, outDims, starts, steps, inStrides, rank);
                return new ConcreteNDArray(out, outShape, DType.FLOAT32, device);
            }
            case F64Storage s -> {
                double[] in = s.data();
                double[] out = new double[size];
                sliceCopyF64(in, out, outDims, starts, steps, inStrides, rank);
                return new ConcreteNDArray(out, outShape, device);
            }
            case I32Storage s -> {
                int[] in = s.data();
                int[] out = new int[size];
                for (int outFlat = 0; outFlat < size; outFlat++) {
                    int inFlat = sliceInIndex(outFlat, outDims, starts, steps, inStrides, rank);
                    out[outFlat] = in[inFlat];
                }
                return new ConcreteNDArray(out, outShape, device);
            }
            case I64Storage s -> {
                long[] in = s.data();
                long[] out = new long[size];
                for (int outFlat = 0; outFlat < size; outFlat++) {
                    int inFlat = sliceInIndex(outFlat, outDims, starts, steps, inStrides, rank);
                    out[outFlat] = in[inFlat];
                }
                return new ConcreteNDArray(out, outShape, device);
            }
            case BoolStorage s -> {
                boolean[] in = s.data();
                boolean[] out = new boolean[size];
                for (int outFlat = 0; outFlat < size; outFlat++) {
                    int inFlat = sliceInIndex(outFlat, outDims, starts, steps, inStrides, rank);
                    out[outFlat] = in[inFlat];
                }
                return new ConcreteNDArray(out, outShape, device);
            }
        }
    }

    private static int sliceInIndex(int outFlat, int[] outDims, int[] starts, int[] steps, int[] inStrides, int rank) {
        int rem = outFlat;
        int inFlat = 0;
        for (int i = rank - 1; i >= 0; i--) {
            int coord = rem % outDims[i];
            rem /= outDims[i];
            inFlat += (starts[i] + coord * steps[i]) * inStrides[i];
        }
        return inFlat;
    }

    private static void sliceCopyF32(float[] in, float[] out, int[] outDims, int[] starts, int[] steps, int[] inStrides, int rank) {
        for (int outFlat = 0; outFlat < out.length; outFlat++) {
            out[outFlat] = in[sliceInIndex(outFlat, outDims, starts, steps, inStrides, rank)];
        }
    }

    private static void sliceCopyF64(double[] in, double[] out, int[] outDims, int[] starts, int[] steps, int[] inStrides, int rank) {
        for (int outFlat = 0; outFlat < out.length; outFlat++) {
            out[outFlat] = in[sliceInIndex(outFlat, outDims, starts, steps, inStrides, rank)];
        }
    }

    @Override
    public NDArray argmax(int axis) {
        if (isTracing()) return toTraced().argmax(axis);
        requireFloatingDtype("argmax");
        return argReduceAxis(axis, true);
    }

    @Override
    public NDArray argmin(int axis) {
        if (isTracing()) return toTraced().argmin(axis);
        requireFloatingDtype("argmin");
        return argReduceAxis(axis, false);
    }

    /**
     * Index of the max/min element along {@code axis}, using the same
     * outer/axis/inner block decomposition as {@link #reduceAxis}, but
     * tracking the winning index within each block instead of accumulating.
     * Returns a real {@code DType.INT32} array regardless of the input's
     * floating width (axis sizes never approach INT32's range).
     */
    private NDArray argReduceAxis(int axis, boolean max) {
        int norm = shape.normalizeAxis(axis);
        int[] dims = shape.dimensions();
        int axisSize = dims[norm];
        int outerSize = 1;
        for (int i = 0; i < norm; i++) outerSize *= dims[i];
        int innerSize = 1;
        for (int i = norm + 1; i < dims.length; i++) innerSize *= dims[i];

        Shape outShape = shape.reduceAxis(norm, false);
        int[] out = new int[(int) outShape.size()];

        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            for (int o = 0; o < outerSize; o++) {
                for (int inr = 0; inr < innerSize; inr++) {
                    int bestIdx = 0;
                    double best = in[o * axisSize * innerSize + inr];
                    for (int a = 1; a < axisSize; a++) {
                        double v = in[o * axisSize * innerSize + a * innerSize + inr];
                        if (max ? v > best : v < best) {
                            best = v;
                            bestIdx = a;
                        }
                    }
                    out[o * innerSize + inr] = bestIdx;
                }
            }
            return new ConcreteNDArray(out, outShape, device);
        }

        float[] in = f32();
        for (int o = 0; o < outerSize; o++) {
            for (int inr = 0; inr < innerSize; inr++) {
                int bestIdx = 0;
                float best = in[o * axisSize * innerSize + inr];
                for (int a = 1; a < axisSize; a++) {
                    float v = in[o * axisSize * innerSize + a * innerSize + inr];
                    if (max ? v > best : v < best) {
                        best = v;
                        bestIdx = a;
                    }
                }
                out[o * innerSize + inr] = bestIdx;
            }
        }
        return new ConcreteNDArray(out, outShape, device);
    }

    /**
     * Applies a unary elementwise op. Requires a floating dtype. Device
     * dispatch (FLOAT32 only) mirrors {@link #elementwise}: on {@link
     * Device#host()} this runs the plain Java loop (no behavior change); on an
     * explicitly placed device it dispatches to that device's {@link
     * ExecutionBackend}. FLOAT64 always runs the host loop.
     */
    private NDArray unary(Primitive primitive, DoubleUnaryOperator op, java.util.function.Function<NDArray, NDArray> tracedOp) {
        if (isTracing()) return tracedOp.apply(toTraced());
        requireFloatingDtype(primitive.toString());

        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            double[] result = new double[in.length];
            if (result.length > PARALLEL_THRESHOLD) {
                IntStream.range(0, result.length).parallel().forEach(i -> result[i] = op.applyAsDouble(in[i]));
            } else {
                for (int i = 0; i < result.length; i++) result[i] = op.applyAsDouble(in[i]);
            }
            return new ConcreteNDArray(result, shape, device);
        }

        float[] in = f32();
        if (!device.equals(Device.host())) {
            float[] result = backendFor(device).unary(primitive, in, device);
            return new ConcreteNDArray(result, shape, dtype, device);
        }

        float[] result = new float[in.length];
        if (result.length > PARALLEL_THRESHOLD) {
            IntStream.range(0, result.length).parallel().forEach(i -> result[i] = (float) op.applyAsDouble(in[i]));
        } else {
            for (int i = 0; i < result.length; i++) result[i] = (float) op.applyAsDouble(in[i]);
        }
        return new ConcreteNDArray(result, shape, dtype, device);
    }

    @Override public NDArray exp() { return unary(Primitive.EXP, Math::exp, NDArray::exp); }
    @Override public NDArray log() { return unary(Primitive.LOG, Math::log, NDArray::log); }
    @Override public NDArray sin() { return unary(Primitive.SIN, Math::sin, NDArray::sin); }
    @Override public NDArray cos() { return unary(Primitive.COS, Math::cos, NDArray::cos); }
    @Override public NDArray sqrt() { return unary(Primitive.SQRT, Math::sqrt, NDArray::sqrt); }
    @Override public NDArray rsqrt() { return unary(Primitive.RSQRT, x -> 1.0 / Math.sqrt(x), NDArray::rsqrt); }

    @Override public NDArray tanh() { return unary(Primitive.TANH, Math::tanh, NDArray::tanh); }
    @Override public NDArray relu() { return unary(Primitive.RELU, x -> Math.max(0.0, x), NDArray::relu); }
    @Override public NDArray sigmoid() { return unary(Primitive.SIGMOID, x -> 1.0 / (1.0 + Math.exp(-x)), NDArray::sigmoid); }

    @Override
    public NDArray reshape(Shape newShape) {
        if (isTracing()) return toTraced().reshape(newShape);
        if (newShape.size() != shape.size()) {
            throw new IllegalArgumentException(
                "reshape: size mismatch " + shape + " -> " + newShape + " (" + shape.size() + " vs " + newShape.size() + ")");
        }
        return switch (storage) {
            case F32Storage s -> new ConcreteNDArray(s.data(), newShape, DType.FLOAT32, device);
            case F64Storage s -> new ConcreteNDArray(s.data(), newShape, device);
            case I32Storage s -> new ConcreteNDArray(s.data(), newShape, device);
            case I64Storage s -> new ConcreteNDArray(s.data(), newShape, device);
            case BoolStorage s -> new ConcreteNDArray(s.data(), newShape, device);
        };
    }

    @Override
    public NDArray transpose(int... axes) {
        if (isTracing()) return toTraced().transpose(axes);
        int rank = shape.rank();
        if (axes.length != rank) {
            throw new IllegalArgumentException(
                "transpose: expected " + rank + " axes, got " + axes.length);
        }
        int[] dims = shape.dimensions();
        int[] outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = dims[axes[i]];
        Shape outShape = new Shape(outDims);
        int[] inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * dims[i + 1];

        int size = (int) shape.size();

        if (dtype == DType.FLOAT32) {
            float[] in = f32();
            float[] out = new float[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int inFlat = transposeIndex(outFlat, outDims, dims, axes, inStrides, rank);
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape, DType.FLOAT32, device);
        }
        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            double[] out = new double[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int inFlat = transposeIndex(outFlat, outDims, dims, axes, inStrides, rank);
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape, device);
        }
        if (dtype == DType.INT32) {
            int[] in = i32();
            int[] out = new int[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int inFlat = transposeIndex(outFlat, outDims, dims, axes, inStrides, rank);
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape, device);
        }
        if (dtype == DType.INT64) {
            long[] in = i64();
            long[] out = new long[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int inFlat = transposeIndex(outFlat, outDims, dims, axes, inStrides, rank);
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape, device);
        }
        if (dtype == DType.BOOL) {
            boolean[] in = bl();
            boolean[] out = new boolean[size];
            for (int outFlat = 0; outFlat < size; outFlat++) {
                int inFlat = transposeIndex(outFlat, outDims, dims, axes, inStrides, rank);
                out[outFlat] = in[inFlat];
            }
            return new ConcreteNDArray(out, outShape, device);
        }
        throw new UnsupportedOperationException("transpose: unsupported dtype " + dtype);
    }

    /** Maps a flat output index to the corresponding flat input index under axis permutation. */
    private static int transposeIndex(int outFlat, int[] outDims, int[] inDims, int[] axes, int[] inStrides, int rank) {
        // Decompose outFlat into outCoords
        int rem = outFlat;
        int[] outCoords = new int[rank];
        for (int i = rank - 1; i >= 0; i--) {
            outCoords[i] = rem % outDims[i];
            rem /= outDims[i];
        }
        // inCoords[axes[i]] = outCoords[i]
        int inFlat = 0;
        for (int i = 0; i < rank; i++) {
            inFlat += outCoords[i] * inStrides[axes[i]];
        }
        return inFlat;
    }

    @Override
    public NDArray pad(int[][] padding) {
        if (isTracing()) return toTraced().pad(padding);
        int rank = shape.rank();
        if (padding.length != rank) {
            throw new IllegalArgumentException(
                "pad: padding must have " + rank + " rows, got " + padding.length);
        }
        int[] inDims = shape.dimensions();
        int[] outDims = new int[rank];
        for (int i = 0; i < rank; i++) {
            outDims[i] = inDims[i] + padding[i][0] + padding[i][1];
        }
        Shape outShape = new Shape(outDims);
        int size = (int) outShape.size();

        // Compute strides for output layout
        int[] outStrides = new int[rank];
        outStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outDims[i + 1];
        int[] inStrides = new int[rank];
        inStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * inDims[i + 1];

        if (dtype == DType.FLOAT32) {
            float[] in = f32();
            float[] out = new float[size]; // zero-initialised
            padCopy(in, out, inDims, outDims, inStrides, outStrides, padding, rank);
            return new ConcreteNDArray(out, outShape, DType.FLOAT32, device);
        }
        if (dtype == DType.FLOAT64) {
            double[] in = f64();
            double[] out = new double[size];
            padCopyD(in, out, inDims, outDims, inStrides, outStrides, padding, rank);
            return new ConcreteNDArray(out, outShape, device);
        }
        if (dtype == DType.INT32) {
            int[] in = i32();
            int[] out = new int[size];
            padCopyI(in, out, inDims, outDims, inStrides, outStrides, padding, rank);
            return new ConcreteNDArray(out, outShape, device);
        }
        throw new UnsupportedOperationException("pad: unsupported dtype " + dtype);
    }

    private static void padCopy(float[] in, float[] out,
                                 int[] inDims, int[] outDims,
                                 int[] inStrides, int[] outStrides,
                                 int[][] padding, int rank) {
        int inSize = 1;
        for (int d : inDims) inSize *= d;
        for (int inFlat = 0; inFlat < inSize; inFlat++) {
            int rem = inFlat;
            int outFlat = 0;
            for (int i = rank - 1; i >= 0; i--) {
                int coord = rem % inDims[i];
                rem /= inDims[i];
                outFlat += (coord + padding[i][0]) * outStrides[i];
            }
            out[outFlat] = in[inFlat];
        }
    }

    private static void padCopyD(double[] in, double[] out,
                                   int[] inDims, int[] outDims,
                                   int[] inStrides, int[] outStrides,
                                   int[][] padding, int rank) {
        int inSize = 1;
        for (int d : inDims) inSize *= d;
        for (int inFlat = 0; inFlat < inSize; inFlat++) {
            int rem = inFlat;
            int outFlat = 0;
            for (int i = rank - 1; i >= 0; i--) {
                int coord = rem % inDims[i];
                rem /= inDims[i];
                outFlat += (coord + padding[i][0]) * outStrides[i];
            }
            out[outFlat] = in[inFlat];
        }
    }

    private static void padCopyI(int[] in, int[] out,
                                  int[] inDims, int[] outDims,
                                  int[] inStrides, int[] outStrides,
                                  int[][] padding, int rank) {
        int inSize = 1;
        for (int d : inDims) inSize *= d;
        for (int inFlat = 0; inFlat < inSize; inFlat++) {
            int rem = inFlat;
            int outFlat = 0;
            for (int i = rank - 1; i >= 0; i--) {
                int coord = rem % inDims[i];
                rem /= inDims[i];
                outFlat += (coord + padding[i][0]) * outStrides[i];
            }
            out[outFlat] = in[inFlat];
        }
    }

    @Override
    public NDArray astype(DType target) {
        if (isTracing()) return toTraced().astype(target);
        if (target == dtype) return this;

        int n = (int) shape.size();
        switch (target) {
            case FLOAT32 -> {
                float[] out = new float[n];
                switch (dtype) {
                    case INT32 -> { int[] in = i32(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    case BOOL -> { boolean[] in = bl(); for (int i = 0; i < n; i++) out[i] = in[i] ? 1f : 0f; }
                    case FLOAT64 -> { double[] in = f64(); for (int i = 0; i < n; i++) out[i] = (float) in[i]; }
                    case INT64 -> { long[] in = i64(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, DType.FLOAT32, device);
            }
            case INT32 -> {
                int[] out = new int[n];
                switch (dtype) {
                    case FLOAT32 -> { float[] in = f32(); for (int i = 0; i < n; i++) out[i] = (int) in[i]; } // truncates toward zero
                    case BOOL -> { boolean[] in = bl(); for (int i = 0; i < n; i++) out[i] = in[i] ? 1 : 0; }
                    case FLOAT64 -> { double[] in = f64(); for (int i = 0; i < n; i++) out[i] = (int) in[i]; } // truncates toward zero
                    case INT64 -> { long[] in = i64(); for (int i = 0; i < n; i++) out[i] = (int) in[i]; } // narrows, may overflow
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case BOOL -> {
                boolean[] out = new boolean[n];
                switch (dtype) {
                    case FLOAT32 -> { float[] in = f32(); for (int i = 0; i < n; i++) out[i] = in[i] != 0f; }
                    case INT32 -> { int[] in = i32(); for (int i = 0; i < n; i++) out[i] = in[i] != 0; }
                    case FLOAT64 -> { double[] in = f64(); for (int i = 0; i < n; i++) out[i] = in[i] != 0.0; }
                    case INT64 -> { long[] in = i64(); for (int i = 0; i < n; i++) out[i] = in[i] != 0L; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case FLOAT64 -> {
                double[] out = new double[n];
                switch (dtype) {
                    case FLOAT32 -> { float[] in = f32(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    case INT32 -> { int[] in = i32(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    case BOOL -> { boolean[] in = bl(); for (int i = 0; i < n; i++) out[i] = in[i] ? 1.0 : 0.0; }
                    case INT64 -> { long[] in = i64(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case INT64 -> {
                long[] out = new long[n];
                switch (dtype) {
                    case FLOAT32 -> { float[] in = f32(); for (int i = 0; i < n; i++) out[i] = (long) in[i]; } // truncates toward zero
                    case INT32 -> { int[] in = i32(); for (int i = 0; i < n; i++) out[i] = in[i]; }
                    case BOOL -> { boolean[] in = bl(); for (int i = 0; i < n; i++) out[i] = in[i] ? 1L : 0L; }
                    case FLOAT64 -> { double[] in = f64(); for (int i = 0; i < n; i++) out[i] = (long) in[i]; } // truncates toward zero
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            default -> throw new IllegalArgumentException("Unsupported astype target: " + target);
        }
    }

    @Override
    public float[] toFloatArray() {
        requireDtype(DType.FLOAT32, "toFloatArray");
        return f32();
    }

    @Override
    public int[] toIntArray() {
        requireDtype(DType.INT32, "toIntArray");
        return i32();
    }

    @Override
    public boolean[] toBoolArray() {
        requireDtype(DType.BOOL, "toBoolArray");
        return bl();
    }

    @Override
    public double[] toDoubleArray() {
        requireDtype(DType.FLOAT64, "toDoubleArray");
        return f64();
    }

    @Override
    public long[] toLongArray() {
        requireDtype(DType.INT64, "toLongArray");
        return i64();
    }

    @Override
    public String toString() {
        return "Array(" + storage.dataString() + ", shape=" + shape + ", dtype=" + dtype + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConcreteNDArray other)) return false;
        if (!shape.equals(other.shape) || dtype != other.dtype) return false;
        return storage.dataEquals(other.storage);
    }

    @Override
    public int hashCode() {
        return storage.dataHashCode() * 31 + shape.hashCode();
    }

    /**
     * Batched matmul over the last two axes with numpy-style batch broadcasting.
     * Public because {@code Grad}'s MATMUL executePrimitive and VJP reuse the same kernel.
     * Both operands must share a floating dtype.
     */
    public static NDArray matmulEager(NDArray a, NDArray b) {
        int[] aDims = a.shape().dimensions();
        int[] bDims = b.shape().dimensions();
        if (aDims.length < 2 || bDims.length < 2) {
            throw new IllegalArgumentException("matmul: both operands must have rank >= 2");
        }
        int M = aDims[aDims.length - 2];
        int Ka = aDims[aDims.length - 1];
        int Kb = bDims[bDims.length - 2];
        int N = bDims[bDims.length - 1];
        if (Ka != Kb) throw new IllegalArgumentException("matmul: inner-dim mismatch " + Ka + " vs " + Kb);
        int K = Ka;

        int[] aBatch = java.util.Arrays.copyOfRange(aDims, 0, aDims.length - 2);
        int[] bBatch = java.util.Arrays.copyOfRange(bDims, 0, bDims.length - 2);
        int[] outBatch = broadcastBatch(aBatch, bBatch);
        int batchTotal = 1;
        for (int d : outBatch) batchTotal *= d;

        int[] outDims = new int[outBatch.length + 2];
        System.arraycopy(outBatch, 0, outDims, 0, outBatch.length);
        outDims[outDims.length - 2] = M;
        outDims[outDims.length - 1] = N;
        Shape outShape = new Shape(outDims);

        int aBatchTotal = 1; for (int d : aBatch) aBatchTotal *= d;
        int bBatchTotal = 1; for (int d : bBatch) bBatchTotal *= d;

        if (a.dtype() == DType.FLOAT64) {
            double[] aData = a.toDoubleArray();
            double[] bData = b.toDoubleArray();
            double[] out = new double[batchTotal * M * N];
            for (int batch = 0; batch < batchTotal; batch++) {
                int aBatchIdx = broadcastBatchIndex(batch, outBatch, aBatch);
                int bBatchIdx = broadcastBatchIndex(batch, outBatch, bBatch);
                int aOff = aBatchIdx * M * K;
                int bOff = bBatchIdx * K * N;
                int outOff = batch * M * N;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        double sum = 0;
                        for (int k = 0; k < K; k++) sum += aData[aOff + i * K + k] * bData[bOff + k * N + j];
                        out[outOff + i * N + j] = sum;
                    }
                }
            }
            return new ConcreteNDArray(out, outShape, a.device());
        }
        float[] aData = a.toFloatArray();
        float[] bData = b.toFloatArray();
        float[] out = new float[batchTotal * M * N];
        for (int batch = 0; batch < batchTotal; batch++) {
            int aBatchIdx = broadcastBatchIndex(batch, outBatch, aBatch);
            int bBatchIdx = broadcastBatchIndex(batch, outBatch, bBatch);
            int aOff = aBatchIdx * M * K;
            int bOff = bBatchIdx * K * N;
            int outOff = batch * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) sum += aData[aOff + i * K + k] * bData[bOff + k * N + j];
                    out[outOff + i * N + j] = sum;
                }
            }
        }
        return new ConcreteNDArray(out, outShape, DType.FLOAT32, a.device());
    }

    /** Broadcasts two batch-dim arrays (numpy right-align) and returns the resulting shape. */
    private static int[] broadcastBatch(int[] a, int[] b) {
        int rank = Math.max(a.length, b.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int da = i < rank - a.length ? 1 : a[i - (rank - a.length)];
            int db = i < rank - b.length ? 1 : b[i - (rank - b.length)];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException(
                    "matmul: incompatible batch shapes " + java.util.Arrays.toString(a) + " and " + java.util.Arrays.toString(b));
            }
            out[i] = Math.max(da, db);
        }
        return out;
    }

    /** Maps an output batch flat index to the corresponding flat index in an operand's batch shape. */
    private static int broadcastBatchIndex(int outFlat, int[] outBatch, int[] opBatch) {
        if (opBatch.length == 0) return 0;
        int rank = outBatch.length;
        int offset = rank - opBatch.length;
        int rem = outFlat;
        int[] coords = new int[rank];
        for (int i = rank - 1; i >= 0; i--) { coords[i] = rem % outBatch[i]; rem /= outBatch[i]; }
        int flat = 0;
        for (int i = 0; i < opBatch.length; i++) {
            int dim = opBatch[i];
            int coord = dim == 1 ? 0 : coords[i + offset];
            flat = flat * dim + coord;
        }
        return flat;
    }
}
