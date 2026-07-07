package com.marmanis.jax4j.ir;

import com.marmanis.jax4j.core.Device;
import java.util.List;

/**
 * Metadata for a {@code PMAP} equation: the body function traced into a
 * single-input/single-output {@link Jaxpr} (per-device shard), and the list of
 * devices the shards run on. {@code numDevices()} == the leading dimension of
 * the equation's input and output {@link Var}s.
 */
public record PmapMeta(Jaxpr bodyJaxpr, List<Device> devices) {
    public int numDevices() { return devices.size(); }
}
