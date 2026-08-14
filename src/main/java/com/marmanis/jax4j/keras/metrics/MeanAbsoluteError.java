package com.marmanis.jax4j.keras.metrics;

import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;

/**
 * Running mean of absolute prediction error.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class MeanAbsoluteError implements Metric {
    private double sum;
    private long count;

    @Override public String name() { return "mae"; }
    @Override public void reset() { sum = 0; count = 0; }

    @Override public void update(NDArray yTrue, NDArray yPred) {
        if (yPred.dtype() == DType.FLOAT64) {
            double[] p = yPred.toDoubleArray();
            double[] t = yTrue.toDoubleArray();
            int n = Math.min(p.length, t.length);
            for (int i = 0; i < n; i++) sum += Math.abs(p[i] - t[i]);
            count += n;
        } else {
            float[] p = yPred.toFloatArray();
            float[] t = yTrue.toFloatArray();
            int n = Math.min(p.length, t.length);
            for (int i = 0; i < n; i++) sum += Math.abs(p[i] - t[i]);
            count += n;
        }
    }

    @Override public float result() { return count == 0 ? 0f : (float) (sum / count); }
}
