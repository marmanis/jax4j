package com.marmanis.jax4j.keras.callbacks;

import com.marmanis.jax4j.keras.Model;

import java.util.Map;

/**
 * Training lifecycle hook, mirroring {@code keras.callbacks.Callback}. {@code logs}
 * is a mutable {@link Map} — callbacks may add entries (e.g. metric snapshots)
 * that other callbacks observe.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public interface Callback {
    default void setModel(Model model) {}
    default void onTrainBegin(Map<String, Float> logs) {}
    default void onTrainEnd(Map<String, Float> logs) {}
    default void onEpochBegin(int epoch, Map<String, Float> logs) {}
    default void onEpochEnd(int epoch, Map<String, Float> logs) {}
    default void onBatchBegin(int batch, Map<String, Float> logs) {}
    default void onBatchEnd(int batch, Map<String, Float> logs) {}
}
