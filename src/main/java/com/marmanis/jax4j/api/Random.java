package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;

/**
 * Explicit-key pseudo-random sampling, mirroring {@code jax.random}. Every
 * sampling call is a pure function of a {@link PRNGKey} and a shape — no
 * hidden global RNG state, so results never depend on call order and the same
 * key always reproduces the same array. {@link #split} derives independent
 * child keys for use in unrelated parts of a computation (one subkey per
 * layer's init, another for a dropout mask, etc.), exactly like
 * {@code jax.random.split}.
 *
 * <p><b>Scope note:</b> the underlying generator is a SplitMix64-style 64-bit
 * mix, not JAX's Threefry counter-based generator — it gets the properties
 * that matter for principled stochastic training (pure, splittable,
 * reproducible, statistically sound) without bit-exact parity with Python JAX.
 *
 * <p><b>Why these are eager, not traced:</b> {@link PRNGKey} is a plain Java
 * value, never an {@link NDArray}, so it never appears inside a {@code Jaxpr}.
 * Every {@code Random} call immediately produces a concrete array. When used
 * inside a function being traced (e.g. {@link Nn#dropout} inside a {@code
 * JAX.grad}'d loss), that array is captured as an ordinary constant input to
 * whatever equation consumes it — the same mechanism {@link Lax}/{@link Nn}
 * already use for closed-over scalars. No new {@code Primitive}, VJP rule, or
 * {@code vmap} batching rule is needed.
 *
 * <p><b>Batched sampling:</b> to get one independent mask/sample per example
 * in a batch (e.g. per-example dropout), sample directly at the batch-inclusive
 * shape — {@code Random.bernoulli(key, p, new Shape(batchSize, ...))} — rather
 * than vmapping over an array of keys.
 */
public final class Random {
    private Random() {}

    private static final long GOLDEN_GAMMA = 0x9E3779B97F4A7C15L;
    private static final long SPLIT_TAG = 0xD1B54A32D192ED03L;
    private static final long SAMPLE_TAG = 0x2545F4914F6CDD1DL;

    /** SplitMix64's finalizer/mixer: a fast, well-mixed avalanche of a 64-bit state. */
    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    /**
     * Derives a domain-separated 64-bit stream value for {@code (state, tag,
     * index)} — different tags (split vs. sample) and different indices never
     * collide, so splitting a key and sampling from it can't accidentally
     * correlate.
     */
    private static long streamValue(long state, long tag, long index) {
        return mix64(state ^ tag ^ (GOLDEN_GAMMA * (index + 1)));
    }

    /** Raw uniform float in {@code [0, 1)} for stream index {@code i}, using the top 24 bits. */
    private static float rawUniform01(long state, long i) {
        long bits = streamValue(state, SAMPLE_TAG, i);
        int top24 = (int) (bits >>> 40); // unsigned shift -> non-negative 24-bit value
        return top24 / (float) (1 << 24);
    }

    // ---- key splitting ----

    /** Mirrors {@code jax.random.split(key, num)}: derives {@code num} independent child keys. */
    public static PRNGKey[] split(PRNGKey key, int num) {
        PRNGKey[] result = new PRNGKey[num];
        for (int i = 0; i < num; i++) {
            result[i] = new PRNGKey(mix64(key.state() ^ SPLIT_TAG ^ (GOLDEN_GAMMA * (i + 1))));
        }
        return result;
    }

    /** Equivalent to {@code split(key, 2)} — the common {@code key, subkey = split(key)} idiom. */
    public static PRNGKey[] split(PRNGKey key) {
        return split(key, 2);
    }

    // ---- sampling ----

    /** Mirrors {@code jax.random.uniform(key, shape, minval=lo, maxval=hi)}. */
    public static NDArray uniform(PRNGKey key, Shape shape, float lo, float hi) {
        int n = (int) shape.size();
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = lo + rawUniform01(key.state(), i) * (hi - lo);
        }
        return new ConcreteNDArray(data, shape);
    }

    /** Equivalent to {@code uniform(key, shape, 0f, 1f)}. */
    public static NDArray uniform(PRNGKey key, Shape shape) {
        return uniform(key, shape, 0f, 1f);
    }

    /**
     * Standard-normal samples via Box–Muller, mirroring {@code jax.random.normal}.
     * Each output element consumes two domain-separated uniform draws.
     */
    public static NDArray normal(PRNGKey key, Shape shape) {
        int n = (int) shape.size();
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            float u1 = Math.max(rawUniform01(key.state(), 2L * i), 1e-7f); // avoid log(0)
            float u2 = rawUniform01(key.state(), 2L * i + 1);
            data[i] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
        }
        return new ConcreteNDArray(data, shape);
    }

    /** Mirrors {@code jax.random.bernoulli(key, p, shape)}: true with probability {@code p}. Returns a real {@code DType.BOOL} array. */
    public static NDArray bernoulli(PRNGKey key, float p, Shape shape) {
        int n = (int) shape.size();
        boolean[] data = new boolean[n];
        for (int i = 0; i < n; i++) {
            data[i] = rawUniform01(key.state(), i) < p;
        }
        return new ConcreteNDArray(data, shape);
    }

    /**
     * A uniformly random permutation of {@code 0..n-1} (Fisher–Yates), mirroring
     * {@code jax.random.permutation(key, n)}. Returns a real {@code DType.INT32}
     * array, consistent with how {@code argmax}/{@code argmin} represent indices.
     */
    public static NDArray permutation(PRNGKey key, int n) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = n - 1; i > 0; i--) {
            float u = rawUniform01(key.state(), i);
            int j = Math.min((int) (u * (i + 1)), i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        return new ConcreteNDArray(idx, new Shape(n));
    }

    // ---- initializers (composed from the above; no new primitives needed) ----

    /**
     * Glorot/Xavier uniform initialization, mirroring common Flax/Haiku
     * initializers: {@code U(-limit, limit)} with {@code limit = sqrt(6 / (fanIn + fanOut))}.
     */
    public static NDArray glorotUniform(PRNGKey key, Shape shape, int fanIn, int fanOut) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        return uniform(key, shape, -limit, limit);
    }

    /** He/Kaiming normal initialization: {@code N(0, 1) * sqrt(2 / fanIn)}. */
    public static NDArray heNormal(PRNGKey key, Shape shape, int fanIn) {
        float std = (float) Math.sqrt(2.0 / fanIn);
        NDArray stdArr = new ConcreteNDArray(new float[]{std}, new Shape(1));
        return normal(key, shape).mul(stdArr);
    }
}
