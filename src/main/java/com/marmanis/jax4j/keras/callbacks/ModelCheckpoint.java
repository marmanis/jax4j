package com.marmanis.jax4j.keras.callbacks;

import com.marmanis.jax4j.keras.Model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Saves the model's parameters to disk at epoch end.
 * If {@code saveBestOnly}, only overwrites when the monitored metric improves.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class ModelCheckpoint implements Callback {

    private final Path path;
    private final String monitor;
    private final boolean saveBestOnly;
    private final boolean lowerIsBetter;

    private float best;
    private Model model;

    public ModelCheckpoint(Path path) { this(path, null, false); }

    public ModelCheckpoint(Path path, String monitor, boolean saveBestOnly) {
        this.path = path;
        this.monitor = monitor;
        this.saveBestOnly = saveBestOnly;
        this.lowerIsBetter = monitor == null || monitor.contains("loss");
        this.best = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
    }

    @Override public void setModel(Model model) { this.model = model; }

    @Override public void onEpochEnd(int epoch, Map<String, Float> logs) {
        boolean save = true;
        if (saveBestOnly && monitor != null) {
            Float cur = logs.get(monitor);
            if (cur == null) return;
            boolean improved = lowerIsBetter ? cur < best : cur > best;
            if (!improved) save = false;
            else best = cur;
        }
        if (save && model != null) {
            try { model.saveWeights(path); }
            catch (IOException e) { throw new RuntimeException("ModelCheckpoint save failed: " + e.getMessage(), e); }
        }
    }
}
