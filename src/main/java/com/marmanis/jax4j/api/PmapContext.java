package com.marmanis.jax4j.api;

/**
 * Thread-local context set on each pmap worker thread while it executes its
 * shard. Gives collective ops ({@code psum}, {@code allGather}) the information
 * they need to coordinate across shards: which device/shard index this thread
 * is, how many total shards there are, and the shared barrier state.
 *
 * <p>Outside a pmap execution this is {@code null}; collective ops treat
 * {@code null} as "identity / not in pmap" and pass through without blocking.
 */
public final class PmapContext {
    private static final ThreadLocal<PmapContext> CURRENT = new ThreadLocal<>();

    final int deviceIndex;
    final int numDevices;
    final SharedCollective collective;

    PmapContext(int deviceIndex, int numDevices, SharedCollective collective) {
        this.deviceIndex = deviceIndex;
        this.numDevices = numDevices;
        this.collective = collective;
    }

    public static PmapContext current() { return CURRENT.get(); }
    static void set(PmapContext ctx)    { CURRENT.set(ctx); }
    static void clear()                 { CURRENT.remove(); }
}
