package com.marmanis.jax4j.backend;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.ir.Primitive;

/**
 * Default execution backend: plain Java loops on the JVM host. Used for
 * {@link Device#host()} and as the fallback target when {@link TornadoVMBackend}
 * can't run (no configured TornadoVM runtime, kernel failure, etc).
 */
public class HostBackend implements ExecutionBackend {
    public static final HostBackend INSTANCE = new HostBackend();

    @Override
    public float[] binary(Primitive primitive, float[] a, float[] b, Device device) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = switch (primitive) {
                case ADD -> a[i] + b[i];
                case SUB -> a[i] - b[i];
                case MUL -> a[i] * b[i];
                case DIV -> a[i] / b[i];
                case GT -> a[i] > b[i] ? 1f : 0f;
                case GE -> a[i] >= b[i] ? 1f : 0f;
                case LT -> a[i] < b[i] ? 1f : 0f;
                case LE -> a[i] <= b[i] ? 1f : 0f;
                case EQ -> a[i] == b[i] ? 1f : 0f;
                case NE -> a[i] != b[i] ? 1f : 0f;
                case MAX -> Math.max(a[i], b[i]);
                case MIN -> Math.min(a[i], b[i]);
                default -> throw new UnsupportedOperationException("Not a binary elementwise primitive: " + primitive);
            };
        }
        return out;
    }

    @Override
    public float[] unary(Primitive primitive, float[] a, Device device) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = switch (primitive) {
                case EXP -> (float) Math.exp(a[i]);
                case LOG -> (float) Math.log(a[i]);
                case SIN -> (float) Math.sin(a[i]);
                case COS -> (float) Math.cos(a[i]);
                case TANH -> (float) Math.tanh(a[i]);
                case RELU -> Math.max(0f, a[i]);
                case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-a[i])));
                default -> throw new UnsupportedOperationException("Not a unary elementwise primitive: " + primitive);
            };
        }
        return out;
    }

    @Override
    public float[] matmul(float[] a, float[] b, int m, int k, int n, Device device) {
        float[] out = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
                out[i * n + j] = sum;
            }
        }
        return out;
    }

    @Override
    public double[] binary(Primitive primitive, double[] a, double[] b, Device device) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = switch (primitive) {
                case ADD -> a[i] + b[i];
                case SUB -> a[i] - b[i];
                case MUL -> a[i] * b[i];
                case DIV -> a[i] / b[i];
                case MAX -> Math.max(a[i], b[i]);
                case MIN -> Math.min(a[i], b[i]);
                default -> throw new UnsupportedOperationException("Not a binary elementwise primitive: " + primitive);
            };
        }
        return out;
    }

    @Override
    public double[] unary(Primitive primitive, double[] a, Device device) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = switch (primitive) {
                case EXP -> Math.exp(a[i]);
                case LOG -> Math.log(a[i]);
                case SIN -> Math.sin(a[i]);
                case COS -> Math.cos(a[i]);
                case TANH -> Math.tanh(a[i]);
                case RELU -> Math.max(0.0, a[i]);
                case SIGMOID -> 1.0 / (1.0 + Math.exp(-a[i]));
                default -> throw new UnsupportedOperationException("Not a unary elementwise primitive: " + primitive);
            };
        }
        return out;
    }

    @Override
    public double[] matmul(double[] a, double[] b, int m, int k, int n, Device device) {
        double[] out = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
                out[i * n + j] = sum;
            }
        }
        return out;
    }

    @Override
    public float[] reduce(Primitive primitive, float[] a, Device device) {
        float sum = 0;
        for (float v : a) sum += v;
        if (primitive == Primitive.MEAN) {
            sum /= a.length;
        }
        return new float[]{sum};
    }

    @Override
    public double[] reduce(Primitive primitive, double[] a, Device device) {
        double sum = 0;
        for (double v : a) sum += v;
        if (primitive == Primitive.MEAN) {
            sum /= a.length;
        }
        return new double[]{sum};
    }
}
