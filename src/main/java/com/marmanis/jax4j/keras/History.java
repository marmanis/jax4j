package com.marmanis.jax4j.keras;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Records per-epoch loss and metric values from {@link Model#fit}. Mirrors
 * {@code keras.callbacks.History}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class History {

    private final Map<String, List<Float>> history = new LinkedHashMap<>();

    /** Adds one value under {@code name} to the running list. */
    public void record(String name, float value) {
        history.computeIfAbsent(name, _ -> new java.util.ArrayList<>()).add(value);
    }

    /** Returns the per-epoch series for {@code name}, or an empty list if none. */
    public List<Float> get(String name) {
        return history.getOrDefault(name, List.of());
    }

    public Map<String, List<Float>> asMap() { return history; }
}
