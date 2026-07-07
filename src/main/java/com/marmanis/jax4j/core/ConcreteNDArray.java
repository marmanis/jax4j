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
 * Storage is genuinely typed: exactly one of {@code floatData}/{@code intData}/
 * {@code boolData}/{@code doubleData}/{@code longData} is populated, matching
 * {@link #dtype()} — there is no "float standing in for bool/int" representation
 * anywhere. Arithmetic (`add`/`mul`/.../`dot`/reductions) requires both operands
 * to share the same <em>floating</em> dtype (FLOAT32 or FLOAT64) and throws
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
    private final float[] floatData;
    private final int[] intData;
    private final boolean[] boolData;
    private final double[] doubleData;
    private final long[] longData;

    private ConcreteNDArray(float[] floatData, int[] intData, boolean[] boolData,
                             double[] doubleData, long[] longData,
                             Shape shape, DType dtype, Device device) {
        this.floatData = floatData;
        this.intData = intData;
        this.boolData = boolData;
        this.doubleData = doubleData;
        this.longData = longData;
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
    }

    public ConcreteNDArray(float[] data, Shape shape, DType dtype, Device device) {
        this(data, null, null, null, null, shape, requireFloat32Dtype(dtype), device);
    }

    private static DType requireFloat32Dtype(DType dtype) {
        if (dtype != DType.FLOAT32) {
            throw new IllegalArgumentException("This constructor only accepts dtype FLOAT32, got " + dtype);
        }
        return dtype;
    }

    public ConcreteNDArray(float[] data, Shape shape) {
        this(data, shape, DType.FLOAT32, Device.defaultDevice());
    }

    public ConcreteNDArray(int[] data, Shape shape, Device device) {
        this(null, data, null, null, null, shape, DType.INT32, device);
    }

    public ConcreteNDArray(int[] data, Shape shape) {
        this(data, shape, Device.defaultDevice());
    }

    public ConcreteNDArray(boolean[] data, Shape shape, Device device) {
        this(null, null, data, null, null, shape, DType.BOOL, device);
    }

    public ConcreteNDArray(boolean[] data, Shape shape) {
        this(data, shape, Device.defaultDevice());
    }

    public ConcreteNDArray(double[] data, Shape shape, Device device) {
        this(null, null, null, data, null, shape, DType.FLOAT64, device);
    }

    public ConcreteNDArray(double[] data, Shape shape) {
        this(data, shape, Device.defaultDevice());
    }

    public ConcreteNDArray(long[] data, Shape shape, Device device) {
        this(null, null, null, null, data, shape, DType.INT64, device);
    }

    public ConcreteNDArray(long[] data, Shape shape) {
        this(data, shape, Device.defaultDevice());
    }

    @Override public Shape shape() { return shape; }
    @Override public DType dtype() { return dtype; }
    @Override public Device device() { return device; }

    @Override
    public NDArray to(Device targetDevice) {
        return new ConcreteNDArray(floatData, intData, boolData, doubleData, longData, shape, dtype, targetDevice);
    }

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
        if (other instanceof TracedNDArray || isTracing()) return tracedOp.apply(toTraced(), other);

        requireSameFloatingDtype(other, primitive.toString());

        if (dtype == DType.FLOAT64) {
            double[] otherData = other.toDoubleArray();
            Shape outShape = Shape.broadcast(shape, other.shape());
            double[] result = new double[(int) outShape.size()];
            Shape thisShape = shape;
            Shape otherShape = other.shape();
            if (result.length > PARALLEL_THRESHOLD) {
                IntStream.range(0, result.length).parallel().forEach(i -> {
                    double v1 = doubleData[thisShape.broadcastIndex(outShape, i)];
                    double v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                    result[i] = op.applyAsDouble(v1, v2);
                });
            } else {
                for (int i = 0; i < result.length; i++) {
                    double v1 = doubleData[thisShape.broadcastIndex(outShape, i)];
                    double v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                    result[i] = op.applyAsDouble(v1, v2);
                }
            }
            return new ConcreteNDArray(result, outShape, device);
        }

        float[] otherData = other.toFloatArray();
        Device otherDevice = other.device();
        boolean hostOnly = device.equals(Device.host()) && otherDevice.equals(Device.host());

        if (!hostOnly) {
            if (!device.equals(otherDevice)) {
                throw new IllegalStateException(
                    "Cannot combine arrays on different devices: " + device + " vs " + otherDevice);
            }
            if (shape.equals(other.shape())) {
                float[] result = backendFor(device).binary(primitive, floatData, otherData, device);
                return new ConcreteNDArray(result, shape, dtype, device);
            }
        }

        Shape outShape = Shape.broadcast(shape, other.shape());
        float[] result = new float[(int) outShape.size()];
        Shape thisShape = shape;
        Shape otherShape = other.shape();

        if (result.length > PARALLEL_THRESHOLD) {
            IntStream.range(0, result.length).parallel().forEach(i -> {
                float v1 = floatData[thisShape.broadcastIndex(outShape, i)];
                float v2 = otherData[otherShape.broadcastIndex(outShape, i)];
                result[i] = (float) op.applyAsDouble(v1, v2);
            });
        } else {
            for (int i = 0; i < result.length; i++) {
                float v1 = floatData[thisShape.broadcastIndex(outShape, i)];
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
     * a long through double would lose precision beyond 2^53.
     */
    private NDArray compareElementwise(NDArray other, Primitive primitive,
                                        java.util.function.BiPredicate<Double, Double> test,
                                        java.util.function.BiFunction<NDArray, NDArray, NDArray> tracedOp) {
        if (other instanceof TracedNDArray || isTracing()) return tracedOp.apply(toTraced(), other);

        if (dtype != other.dtype()) {
            throw new IllegalArgumentException(
                primitive + " requires both operands to share a dtype, got " + dtype + " and " + other.dtype()
                    + " — use .astype() to convert.");
        }

        Shape outShape = Shape.broadcast(shape, other.shape());
        boolean[] result = new boolean[(int) outShape.size()];

        switch (dtype) {
            case FLOAT32 -> {
                float[] otherData = other.toFloatArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = floatData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case FLOAT64 -> {
                double[] otherData = other.toDoubleArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = doubleData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case INT32 -> {
                // Compared as double (lossless for int32, unlike a float cast) to
                // avoid precision loss for indices beyond float32's 24-bit mantissa.
                int[] otherData = other.toIntArray();
                for (int i = 0; i < result.length; i++) {
                    double v1 = intData[shape.broadcastIndex(outShape, i)];
                    double v2 = otherData[other.shape().broadcastIndex(outShape, i)];
                    result[i] = test.test(v1, v2);
                }
            }
            case INT64 -> {
                long[] otherData = other.toLongArray();
                for (int i = 0; i < result.length; i++) {
                    long v1 = longData[shape.broadcastIndex(outShape, i)];
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
            default -> throw new IllegalArgumentException(
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
        if (other instanceof TracedNDArray) return toTraced().dot(other);
        if (isTracing()) return toTraced().dot(other);
        requireSameFloatingDtype(other, "dot");
        int M = shape.dimensions()[0];
        int K = shape.dimensions()[1];
        int N = other.shape().dimensions()[1];

        if (dtype == DType.FLOAT64) {
            double[] otherData = other.toDoubleArray();
            double[] result = new double[M * N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0;
                    for (int k = 0; k < K; k++) sum += doubleData[i * K + k] * otherData[k * N + j];
                    result[i * N + j] = sum;
                }
            }
            return new ConcreteNDArray(result, new Shape(M, N), device);
        }

        float[] otherData = other.toFloatArray();
        Device otherDevice = other.device();

        boolean hostOnly = device.equals(Device.host()) && otherDevice.equals(Device.host());
        if (!hostOnly) {
            if (!device.equals(otherDevice)) {
                throw new IllegalStateException(
                    "Cannot combine arrays on different devices: " + device + " vs " + otherDevice);
            }
            float[] result = backendFor(device).matmul(floatData, otherData, M, K, N, device);
            return new ConcreteNDArray(result, new Shape(M, N), dtype, device);
        }

        float[] result = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += floatData[i * K + k] * otherData[k * N + j];
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
            for (double v : doubleData) sum += v;
            return new ConcreteNDArray(new double[]{sum}, new Shape(1), device);
        }
        float sum = 0;
        for (float v : floatData) sum += v;
        return new ConcreteNDArray(new float[]{sum}, new Shape(1), dtype, device);
    }

    @Override
    public NDArray mean() {
        if (isTracing()) return toTraced().mean();
        requireFloatingDtype("mean");
        if (dtype == DType.FLOAT64) {
            double sum = 0;
            for (double v : doubleData) sum += v;
            return new ConcreteNDArray(new double[]{sum / doubleData.length}, new Shape(1), device);
        }
        float sum = 0;
        for (float v : floatData) sum += v;
        return new ConcreteNDArray(new float[]{sum / floatData.length}, new Shape(1), dtype, device);
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
            double[] out = new double[(int) outShape.size()];
            for (int o = 0; o < outerSize; o++) {
                for (int in = 0; in < innerSize; in++) {
                    double total = 0;
                    for (int a = 0; a < axisSize; a++) {
                        total += doubleData[o * axisSize * innerSize + a * innerSize + in];
                    }
                    out[o * innerSize + in] = mean ? total / axisSize : total;
                }
            }
            return new ConcreteNDArray(out, outShape, device);
        }

        float[] out = new float[(int) outShape.size()];
        for (int o = 0; o < outerSize; o++) {
            for (int in = 0; in < innerSize; in++) {
                float total = 0;
                for (int a = 0; a < axisSize; a++) {
                    total += floatData[o * axisSize * innerSize + a * innerSize + in];
                }
                out[o * innerSize + in] = mean ? total / axisSize : total;
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
            for (int o = 0; o < outerSize; o++) {
                for (int in = 0; in < innerSize; in++) {
                    int bestIdx = 0;
                    double best = doubleData[o * axisSize * innerSize + in];
                    for (int a = 1; a < axisSize; a++) {
                        double v = doubleData[o * axisSize * innerSize + a * innerSize + in];
                        if (max ? v > best : v < best) {
                            best = v;
                            bestIdx = a;
                        }
                    }
                    out[o * innerSize + in] = bestIdx;
                }
            }
            return new ConcreteNDArray(out, outShape, device);
        }

        for (int o = 0; o < outerSize; o++) {
            for (int in = 0; in < innerSize; in++) {
                int bestIdx = 0;
                float best = floatData[o * axisSize * innerSize + in];
                for (int a = 1; a < axisSize; a++) {
                    float v = floatData[o * axisSize * innerSize + a * innerSize + in];
                    if (max ? v > best : v < best) {
                        best = v;
                        bestIdx = a;
                    }
                }
                out[o * innerSize + in] = bestIdx;
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
            double[] result = new double[doubleData.length];
            if (result.length > PARALLEL_THRESHOLD) {
                IntStream.range(0, result.length).parallel().forEach(i -> result[i] = op.applyAsDouble(doubleData[i]));
            } else {
                for (int i = 0; i < result.length; i++) result[i] = op.applyAsDouble(doubleData[i]);
            }
            return new ConcreteNDArray(result, shape, device);
        }

        if (!device.equals(Device.host())) {
            float[] result = backendFor(device).unary(primitive, floatData, device);
            return new ConcreteNDArray(result, shape, dtype, device);
        }

        float[] result = new float[floatData.length];
        if (result.length > PARALLEL_THRESHOLD) {
            IntStream.range(0, result.length).parallel().forEach(i -> result[i] = (float) op.applyAsDouble(floatData[i]));
        } else {
            for (int i = 0; i < result.length; i++) result[i] = (float) op.applyAsDouble(floatData[i]);
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
                    case INT32 -> { for (int i = 0; i < n; i++) out[i] = intData[i]; }
                    case BOOL -> { for (int i = 0; i < n; i++) out[i] = boolData[i] ? 1f : 0f; }
                    case FLOAT64 -> { for (int i = 0; i < n; i++) out[i] = (float) doubleData[i]; }
                    case INT64 -> { for (int i = 0; i < n; i++) out[i] = longData[i]; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, DType.FLOAT32, device);
            }
            case INT32 -> {
                int[] out = new int[n];
                switch (dtype) {
                    case FLOAT32 -> { for (int i = 0; i < n; i++) out[i] = (int) floatData[i]; } // truncates toward zero
                    case BOOL -> { for (int i = 0; i < n; i++) out[i] = boolData[i] ? 1 : 0; }
                    case FLOAT64 -> { for (int i = 0; i < n; i++) out[i] = (int) doubleData[i]; } // truncates toward zero
                    case INT64 -> { for (int i = 0; i < n; i++) out[i] = (int) longData[i]; } // narrows, may overflow
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case BOOL -> {
                boolean[] out = new boolean[n];
                switch (dtype) {
                    case FLOAT32 -> { for (int i = 0; i < n; i++) out[i] = floatData[i] != 0f; }
                    case INT32 -> { for (int i = 0; i < n; i++) out[i] = intData[i] != 0; }
                    case FLOAT64 -> { for (int i = 0; i < n; i++) out[i] = doubleData[i] != 0.0; }
                    case INT64 -> { for (int i = 0; i < n; i++) out[i] = longData[i] != 0L; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case FLOAT64 -> {
                double[] out = new double[n];
                switch (dtype) {
                    case FLOAT32 -> { for (int i = 0; i < n; i++) out[i] = floatData[i]; }
                    case INT32 -> { for (int i = 0; i < n; i++) out[i] = intData[i]; }
                    case BOOL -> { for (int i = 0; i < n; i++) out[i] = boolData[i] ? 1.0 : 0.0; }
                    case INT64 -> { for (int i = 0; i < n; i++) out[i] = longData[i]; }
                    default -> throw new IllegalStateException("Unsupported cast " + dtype + " -> " + target);
                }
                return new ConcreteNDArray(out, shape, device);
            }
            case INT64 -> {
                long[] out = new long[n];
                switch (dtype) {
                    case FLOAT32 -> { for (int i = 0; i < n; i++) out[i] = (long) floatData[i]; } // truncates toward zero
                    case INT32 -> { for (int i = 0; i < n; i++) out[i] = intData[i]; }
                    case BOOL -> { for (int i = 0; i < n; i++) out[i] = boolData[i] ? 1L : 0L; }
                    case FLOAT64 -> { for (int i = 0; i < n; i++) out[i] = (long) doubleData[i]; } // truncates toward zero
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
        return floatData;
    }

    @Override
    public int[] toIntArray() {
        requireDtype(DType.INT32, "toIntArray");
        return intData;
    }

    @Override
    public boolean[] toBoolArray() {
        requireDtype(DType.BOOL, "toBoolArray");
        return boolData;
    }

    @Override
    public double[] toDoubleArray() {
        requireDtype(DType.FLOAT64, "toDoubleArray");
        return doubleData;
    }

    @Override
    public long[] toLongArray() {
        requireDtype(DType.INT64, "toLongArray");
        return longData;
    }

    @Override
    public String toString() {
        return "Array(" + activeDataString() + ", shape=" + shape + ", dtype=" + dtype + ")";
    }

    private String activeDataString() {
        return switch (dtype) {
            case FLOAT32 -> Arrays.toString(floatData);
            case INT32 -> Arrays.toString(intData);
            case BOOL -> Arrays.toString(boolData);
            case FLOAT64 -> Arrays.toString(doubleData);
            case INT64 -> Arrays.toString(longData);
            default -> "<unsupported dtype " + dtype + ">";
        };
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConcreteNDArray other)) return false;
        if (!shape.equals(other.shape) || dtype != other.dtype) return false;
        return switch (dtype) {
            case FLOAT32 -> Arrays.equals(floatData, other.floatData);
            case INT32 -> Arrays.equals(intData, other.intData);
            case BOOL -> Arrays.equals(boolData, other.boolData);
            case FLOAT64 -> Arrays.equals(doubleData, other.doubleData);
            case INT64 -> Arrays.equals(longData, other.longData);
            default -> false;
        };
    }

    @Override
    public int hashCode() {
        int dataHash = switch (dtype) {
            case FLOAT32 -> Arrays.hashCode(floatData);
            case INT32 -> Arrays.hashCode(intData);
            case BOOL -> Arrays.hashCode(boolData);
            case FLOAT64 -> Arrays.hashCode(doubleData);
            case INT64 -> Arrays.hashCode(longData);
            default -> 0;
        };
        return dataHash * 31 + shape.hashCode();
    }
}
