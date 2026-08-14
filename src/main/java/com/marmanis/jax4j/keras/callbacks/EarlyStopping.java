package com.marmanis.jax4j.keras.callbacks;

import com.marmanis.jax4j.keras.Model;

import java.util.Map;

/**
 * Stops training when a monitored metric stops improving.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class EarlyStopping implements Callback {

    private final String monitor;
    private final int patience;
    private final float minDelta;
    private final boolean lowerIsBetter;

    private float best;
    private int wait;
    private Model model;

    public EarlyStopping(String monitor, int patience, float minDelta) {
        this.monitor = monitor;
        this.patience = patience;
        this.minDelta = minDelta;
        this.lowerIsBetter = monitor.contains("loss");
        this.best = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        this.wait = 0;
    }

    @Override public void setModel(Model model) { this.model = model; }

    @Override public void onEpochEnd(int epoch, Map<String, Float> logs) {
        Float current = logs.get(monitor);
        if (current == null) return;
        boolean improved = lowerIsBetter
            ? current < best - minDelta
            : current > best + minDelta;
        if (improved) {
            best = current;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience && model != null) {
                model.stopTraining(true);
            }
        }
    }
}
