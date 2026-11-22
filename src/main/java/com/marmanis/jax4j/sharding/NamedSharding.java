package com.marmanis.jax4j.sharding;

import java.util.Objects;

/**
 * A sharding strategy that maps tensor dimensions to a {@link DeviceMesh} using a {@link PartitionSpec}.
 * Mirrors {@code jax.sharding.NamedSharding}.
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record NamedSharding(DeviceMesh mesh, PartitionSpec spec) {

    public NamedSharding {
        Objects.requireNonNull(mesh, "mesh must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
    }

    public static NamedSharding of(DeviceMesh mesh, PartitionSpec spec) {
        return new NamedSharding(mesh, spec);
    }

    public static NamedSharding replicated(DeviceMesh mesh) {
        return new NamedSharding(mesh, PartitionSpec.replicated());
    }
}
