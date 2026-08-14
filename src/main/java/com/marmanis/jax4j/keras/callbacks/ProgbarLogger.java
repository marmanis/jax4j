package com.marmanis.jax4j.keras.callbacks;

import java.util.Map;

/**
 * Prints a per-epoch summary line: {@code Epoch 3/5 - loss=0.1234 accuracy=0.9012}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class ProgbarLogger implements Callback {

    private int totalEpochs = -1;

    /** Called by {@link com.marmanis.jax4j.keras.Model#fit} before training starts. */
    @Override public void onTrainBegin(Map<String, Float> logs) {
        Float t = logs == null ? null : logs.get("epochs");
        totalEpochs = t == null ? -1 : t.intValue();
    }

    @Override public void onEpochEnd(int epoch, Map<String, Float> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("Epoch ").append(epoch + 1);
        if (totalEpochs > 0) sb.append("/").append(totalEpochs);
        sb.append(" -");
        for (Map.Entry<String, Float> e : logs.entrySet()) {
            if (e.getKey().equals("epochs")) continue;
            sb.append(" ").append(e.getKey()).append("=").append(String.format("%.4f", e.getValue()));
        }
        System.out.println(sb);
    }
}
