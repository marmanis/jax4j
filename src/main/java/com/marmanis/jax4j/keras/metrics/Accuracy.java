package com.marmanis.jax4j.keras.metrics;

import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;

/**
 * Classification accuracy: fraction of examples whose predicted class (argmax
 * of the last axis of {@code yPred}) matches the true class. Accepts targets
 * either as class-index integers or as one-hot vectors of the same rank as
 * {@code yPred} (in which case argmax is also applied to {@code yTrue}).
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Accuracy implements Metric {
    private long correct;
    private long total;

    @Override public String name() { return "accuracy"; }

    @Override public void reset() { correct = 0; total = 0; }

    @Override public void update(NDArray yTrue, NDArray yPred) {
        int[] predIdx = yPred.argmax(-1).toIntArray();
        int[] trueIdx;
        if (yTrue.shape().rank() == yPred.shape().rank()) {
            trueIdx = (yTrue.dtype() == DType.INT32 || yTrue.dtype() == DType.INT64)
                ? flatInts(yTrue)
                : yTrue.argmax(-1).toIntArray();
        } else {
            trueIdx = flatInts(yTrue);
        }
        int n = Math.min(predIdx.length, trueIdx.length);
        for (int i = 0; i < n; i++) if (predIdx[i] == trueIdx[i]) correct++;
        total += n;
    }

    private static int[] flatInts(NDArray a) {
        if (a.dtype() == DType.INT32) return a.toIntArray();
        if (a.dtype() == DType.INT64) {
            long[] l = a.toLongArray();
            int[] out = new int[l.length];
            for (int i = 0; i < l.length; i++) out[i] = (int) l[i];
            return out;
        }
        float[] f = a.toFloatArray();
        int[] out = new int[f.length];
        for (int i = 0; i < f.length; i++) out[i] = (int) f[i];
        return out;
    }

    @Override public float result() { return total == 0 ? 0f : (float) ((double) correct / total); }
}
