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
        if (dtype == DType.FLOAT64) {
            double sum = 0;
            for (double v : f64()) sum += v;
            return new ConcreteNDArray(new double[]{sum}, new Shape(1), device);
        }
        float sum = 0;
        for (float v : f32()) sum += v;
        return new ConcreteNDArray(new float[]{sum}, new Shape(1), dtype, device);
    }

    @Override
    public NDArray mean() {
        if (isTracing()) return toTraced().mean();
        requireFloatingDtype("mean");
        if (dtype == DType.FLOAT64) {
            double[] data = f64();
            double sum = 0;
            for (double v : data) sum += v;
            return new ConcreteNDArray(new double[]{sum / data.length}, new Shape(1), device);
        }
        float[] data = f32();
        float sum = 0;
        for (float v : data) sum += v;
        return new ConcreteNDArray(new float[]{sum / data.length}, new Shape(1), dtype, device);
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
            double[] out = new double[(int) outShape.size()];
            for (int o = 0; o < outerSize; o++) {
                for (int inr = 0; inr < innerSize; inr++) {
                    double total = 0;
                    for (int a = 0; a < axisSize; a++) {
                        total += in[o * axisSize * innerSize + a * innerSize + inr];
                    }
                    out[o * innerSize + inr] = mean ? total / axisSize : total;
                }
            }
            return new ConcreteNDArray(out, outShape, device);
        }

        float[] in = f32();
        float[] out = new float[(int) outShape.size()];
        for (int o = 0; o < outerSize; o++) {
            for (int inr = 0; inr < innerSize; inr++) {
                float total = 0;
                for (int a = 0; a < axisSize; a++) {
                    total += in[o * axisSize * innerSize + a * innerSize + inr];
                }
                out[o * innerSize + inr] = mean ? total / axisSize : total;
            }
        }
        return new ConcreteNDArray(out, outShape, dtype, device);
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

    @Override public NDArray tanh() { return unary(Primitive.TANH, Math::tanh, NDArray::tanh); }
    @Override public NDArray relu() { return unary(Primitive.RELU, x -> Math.max(0.0, x), NDArray::relu); }
    @Override public NDArray sigmoid() { return unary(Primitive.SIGMOID, x -> 1.0 / (1.0 + Math.exp(-x)), NDArray::sigmoid); }

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
}
