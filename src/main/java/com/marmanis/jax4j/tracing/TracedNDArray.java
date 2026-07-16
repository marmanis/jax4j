package com.marmanis.jax4j.tracing;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;

import java.util.List;

/**
 * An NDArray that records its operations into a Tracer instead of performing them eagerly.
 */
public class TracedNDArray implements NDArray {
    private final Var var;

    public TracedNDArray(Var var) {
        this.var = var;
    }

    public Var getVar() { return var; }

    @Override public Shape shape() { return var.shape(); }
    @Override public DType dtype() { return var.dtype(); }
    @Override public Device device() { return Device.defaultDevice(); }

    /**
     * No-op: traced values are symbolic placeholders with no underlying data to
     * move, and {@code grad}'s backward pass always executes on the host
     * regardless of the device tag a concrete leaf was placed on.
     */
    @Override public NDArray to(Device device) { return this; }

    private Var toVar(NDArray a) {
        if (a instanceof TracedNDArray) return ((TracedNDArray) a).getVar();
        return Tracer.current().nextConstant(a);
    }

    private NDArray applyPrimitive(Primitive primitive, List<NDArray> inputs, Shape outShape) {
        return applyPrimitive(primitive, inputs, outShape, dtype(), null);
    }

    private NDArray applyPrimitive(Primitive primitive, List<NDArray> inputs, Shape outShape, Object metadata) {
        return applyPrimitive(primitive, inputs, outShape, dtype(), metadata);
    }

    /**
     * Records an equation whose output dtype differs from the receiver's own
     * dtype — needed by comparisons (always BOOL), {@code argmax}/{@code argmin}
     * (always INT32), and {@code astype} (the cast target).
     */
    private NDArray applyPrimitive(Primitive primitive, List<NDArray> inputs, Shape outShape, DType outDtype, Object metadata) {
        Tracer tracer = Tracer.current();
        List<Var> inVars = inputs.stream().map(this::toVar).toList();
        Var outVar = tracer.nextVar(outShape, outDtype);
        tracer.addEquation(new Equation(inVars, List.of(outVar), primitive, metadata));
        return new TracedNDArray(outVar);
    }

    private NDArray elementwise(NDArray other, Primitive primitive, java.util.function.BiFunction<NDArray, NDArray, NDArray> tracedOp) {
        if (dtype() != other.dtype()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            return tracedOp.apply(left, right);
        }
        return applyPrimitive(primitive, List.of(this, other), Shape.broadcast(shape(), other.shape()));
    }

    private NDArray compareElementwise(NDArray other, Primitive primitive, DType outDtype, java.util.function.BiFunction<NDArray, NDArray, NDArray> tracedOp) {
        if (dtype() != other.dtype()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            return tracedOp.apply(left, right);
        }
        return applyPrimitive(primitive, List.of(this, other), Shape.broadcast(shape(), other.shape()), outDtype, null);
    }

    @Override public NDArray add(NDArray other) { return elementwise(other, Primitive.ADD, NDArray::add); }
    @Override public NDArray sub(NDArray other) { return elementwise(other, Primitive.SUB, NDArray::sub); }
    @Override public NDArray mul(NDArray other) { return elementwise(other, Primitive.MUL, NDArray::mul); }
    @Override public NDArray div(NDArray other) { return elementwise(other, Primitive.DIV, NDArray::div); }

    @Override
    public NDArray dot(NDArray other) {
        if (dtype() != other.dtype()) {
            DType target = DType.promote(this.dtype(), other.dtype());
            NDArray left = (this.dtype() == target) ? this : this.astype(target);
            NDArray right = (other.dtype() == target) ? other : other.astype(target);
            return left.dot(right);
        }
        Shape outShape = new Shape(shape().dimensions()[0], other.shape().dimensions()[1]);
        return applyPrimitive(Primitive.DOT, List.of(this, other), outShape);
    }

    @Override public NDArray exp() { return applyPrimitive(Primitive.EXP, List.of(this), shape()); }
    @Override public NDArray log() { return applyPrimitive(Primitive.LOG, List.of(this), shape()); }
    @Override public NDArray sin() { return applyPrimitive(Primitive.SIN, List.of(this), shape()); }
    @Override public NDArray cos() { return applyPrimitive(Primitive.COS, List.of(this), shape()); }

    @Override public NDArray tanh() { return applyPrimitive(Primitive.TANH, List.of(this), shape()); }
    @Override public NDArray relu() { return applyPrimitive(Primitive.RELU, List.of(this), shape()); }
    @Override public NDArray sigmoid() { return applyPrimitive(Primitive.SIGMOID, List.of(this), shape()); }

    @Override public NDArray sum() { return applyPrimitive(Primitive.SUM, List.of(this), new Shape(1)); }
    @Override public NDArray mean() { return applyPrimitive(Primitive.MEAN, List.of(this), new Shape(1)); }

    @Override
    public NDArray sum(int axis, boolean keepDims) {
        int norm = shape().normalizeAxis(axis);
        return applyPrimitive(Primitive.SUM_AXIS, List.of(this), shape().reduceAxis(norm, keepDims), new AxisMeta(norm, keepDims));
    }

    @Override
    public NDArray mean(int axis, boolean keepDims) {
        int norm = shape().normalizeAxis(axis);
        return applyPrimitive(Primitive.MEAN_AXIS, List.of(this), shape().reduceAxis(norm, keepDims), new AxisMeta(norm, keepDims));
    }

    @Override public NDArray gt(NDArray other) { return compareElementwise(other, Primitive.GT, DType.BOOL, NDArray::gt); }
    @Override public NDArray ge(NDArray other) { return compareElementwise(other, Primitive.GE, DType.BOOL, NDArray::ge); }
    @Override public NDArray lt(NDArray other) { return compareElementwise(other, Primitive.LT, DType.BOOL, NDArray::lt); }
    @Override public NDArray le(NDArray other) { return compareElementwise(other, Primitive.LE, DType.BOOL, NDArray::le); }
    @Override public NDArray eq(NDArray other) { return compareElementwise(other, Primitive.EQ, DType.BOOL, NDArray::eq); }
    @Override public NDArray ne(NDArray other) { return compareElementwise(other, Primitive.NE, DType.BOOL, NDArray::ne); }

    @Override public NDArray max(NDArray other) { return elementwise(other, Primitive.MAX, NDArray::max); }
    @Override public NDArray min(NDArray other) { return elementwise(other, Primitive.MIN, NDArray::min); }

    @Override
    public NDArray argmax(int axis) {
        int norm = shape().normalizeAxis(axis);
        return applyPrimitive(Primitive.ARGMAX, List.of(this), shape().reduceAxis(norm, false), DType.INT32, new AxisMeta(norm, false));
    }

    @Override
    public NDArray argmin(int axis) {
        int norm = shape().normalizeAxis(axis);
        return applyPrimitive(Primitive.ARGMIN, List.of(this), shape().reduceAxis(norm, false), DType.INT32, new AxisMeta(norm, false));
    }

    @Override
    public NDArray astype(DType target) {
        return applyPrimitive(Primitive.CAST, List.of(this), shape(), target, target);
    }

    /**
     * Records an FFI call into the Jaxpr.
     */
    public NDArray applyFFI(String targetName, Shape outShape, NDArray[] inputs) {
        Tracer tracer = Tracer.current();
        List<Var> inVars = java.util.Arrays.stream(inputs).map(this::toVar).toList();
        Var outVar = tracer.nextVar(outShape, dtype());
        
        tracer.addEquation(new Equation(inVars, List.of(outVar), Primitive.FFI_CALL, targetName));
        return new TracedNDArray(outVar);
    }

    @Override
    public float[] toFloatArray() {
        throw new UnsupportedOperationException("Cannot get data from a traced array. You must JIT compile it first.");
    }

    @Override
    public int[] toIntArray() {
        throw new UnsupportedOperationException("Cannot get data from a traced array. You must JIT compile it first.");
    }

    @Override
    public boolean[] toBoolArray() {
        throw new UnsupportedOperationException("Cannot get data from a traced array. You must JIT compile it first.");
    }

    @Override
    public double[] toDoubleArray() {
        throw new UnsupportedOperationException("Cannot get data from a traced array. You must JIT compile it first.");
    }

    @Override
    public long[] toLongArray() {
        throw new UnsupportedOperationException("Cannot get data from a traced array. You must JIT compile it first.");
    }
}
