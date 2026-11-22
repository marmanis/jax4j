package com.marmanis.jax4j.sharding;

import java.util.Arrays;
import java.util.Objects;

/**
 * Specifies how tensor dimensions map to named mesh axes, mirroring {@code jax.sharding.PartitionSpec}.
 *
 * <p>Each element corresponds to an array dimension:
 * <ul>
 *   <li>{@code "data"} or a mesh axis name: the dimension is sharded across that mesh axis.</li>
 *   <li>{@code null}: the dimension is replicated / unpartitioned across the mesh.</li>
 * </ul>
 *
 * <p>For example, {@code PartitionSpec.of("data", null)} for a 2-D matrix shards rows across the "data"
 * devices while replicating columns.
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class PartitionSpec {

    private final String[] partitions;

    public PartitionSpec(String... partitions) {
        this.partitions = partitions == null ? new String[0] : partitions.clone();
    }

    public static PartitionSpec of(String... partitions) {
        return new PartitionSpec(partitions);
    }

    /** Fully replicated across all mesh axes. */
    public static PartitionSpec replicated() {
        return new PartitionSpec();
    }

    public String[] partitions() {
        return partitions.clone();
    }

    public int rank() {
        return partitions.length;
    }

    public String get(int dim) {
        if (dim >= partitions.length) return null;
        return partitions[dim];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof PartitionSpec that)) return false;
        return Arrays.equals(partitions, that.partitions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(partitions);
    }

    @Override
    public String toString() {
        return "PartitionSpec" + Arrays.toString(partitions);
    }
}
