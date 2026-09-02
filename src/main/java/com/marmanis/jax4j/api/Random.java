package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.RandomMeta;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.ArrayList;
import java.util.List;

/**
 * Explicit-key pseudo-random sampling, mirroring {@code jax.random}. Every
 * sampling call is a pure function of a {@link PRNGKey} and a shape — no
 * hidden global RNG state, so results never depend on call order and the same
 * key always reproduces the same array. {@link #split} derives independent
 * child keys for use in unrelated parts of a computation (one subkey per
 * layer's init, another for a dropout mask, etc.), exactly like
 * {@code jax.random.split}.
 *
 * <p>Supports both eager execution and tracing into a {@code Jaxpr} for JIT compilation.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
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
        if (key.keyArray() instanceof TracedNDArray traced) {
            Tracer tracer = Tracer.current();
            List<Var> outVars = new ArrayList<>();
            PRNGKey[] result = new PRNGKey[num];
            for (int i = 0; i < num; i++) {
                Var outVar = tracer.nextVar(new Shape(), DType.INT64);
                outVars.add(outVar);
                result[i] = new PRNGKey(new TracedNDArray(outVar));
            }
            tracer.addEquation(new Equation(
                List.of(traced.getVar()),
                outVars,
                Primitive.RANDOM_SPLIT
            ));
            return result;
        } else {
            return splitEager(key.keyArray(), num);
        }
    }

    public static PRNGKey[] splitEager(NDArray key, int num) {
        long state = key.toLongArray()[0];
        PRNGKey[] result = new PRNGKey[num];
        for (int i = 0; i < num; i++) {
            long subState = mix64(state ^ SPLIT_TAG ^ (GOLDEN_GAMMA * (i + 1)));
            result[i] = new PRNGKey(new ConcreteNDArray(new long[]{subState}, new Shape()));
        }
        return result;
    }

    /** Equivalent to {@code split(key, 2)} — the common {@code key, subkey = split(key)} idiom. */
    public static PRNGKey[] split(PRNGKey key) {
        return split(key, 2);
    }

    /**
     * Derive one subkey from a key and a 64-bit index, mirroring
     * {@code jax.random.fold_in}. Domain-separated from {@link #split} via
     * a distinct tag, so mixing {@code foldIn} and {@code split} on the same
     * root key never accidentally correlates. Only the eager form is provided
     * today; passing a traced key raises {@link UnsupportedOperationException}
     * so callers know to derive per-batch keys on the host side.
     *
     * <p>This is the one-key analogue of {@code split(key, n)[i]} but avoids
     * allocating {@code n} keys just to keep one — cheaper when threading
     * a fresh subkey per training step or per layer within a model.
     */
    public static PRNGKey foldIn(PRNGKey key, long index) {
        if (key.keyArray() instanceof TracedNDArray) {
            throw new UnsupportedOperationException(
                "Random.foldIn on a traced key is not yet supported; " +
                "derive the subkey on the host side and pass it into the traced function as an input.");
        }
        long state = key.keyArray().toLongArray()[0];
        long subState = mix64(state ^ SPLIT_TAG ^ (GOLDEN_GAMMA * (index + 1)));
        return new PRNGKey(new ConcreteNDArray(new long[]{subState}, new Shape()));
    }

    // ---- sampling ----

    /** Mirrors {@code jax.random.uniform(key, shape, minval=lo, maxval=hi)}. */
    public static NDArray uniform(PRNGKey key, Shape shape, float lo, float hi) {
        if (key.keyArray() instanceof TracedNDArray traced) {
            Tracer tracer = Tracer.current();
            Var outVar = tracer.nextVar(shape, DType.FLOAT32);
            tracer.addEquation(new Equation(
                List.of(traced.getVar()),
                List.of(outVar),
                Primitive.RANDOM_UNIFORM,
                new RandomMeta.Uniform(shape, lo, hi)
            ));
            return new TracedNDArray(outVar);
        } else {
            return uniformEager(key.keyArray(), shape, lo, hi);
        }
    }

    public static NDArray uniformEager(NDArray key, Shape shape, float lo, float hi) {
        long state = key.toLongArray()[0];
        int n = (int) shape.size();
        float[] data = new float[n];
        // Each element is a pure function of (state, i), so the loop is
        // trivially parallel. Big kernels (millions of samples in ResNet /
        // DenseNet init) go from single-threaded 100ms-per-layer to
        // amortized microseconds.
        if (n >= PARALLEL_INIT_THRESHOLD) {
            java.util.stream.IntStream.range(0, n).parallel()
                    .forEach(i -> data[i] = lo + rawUniform01(state, i) * (hi - lo));
        } else {
            for (int i = 0; i < n; i++) data[i] = lo + rawUniform01(state, i) * (hi - lo);
        }
        return new ConcreteNDArray(data, shape);
    }

    private static final int PARALLEL_INIT_THRESHOLD = 16_384;

    /** Equivalent to {@code uniform(key, shape, 0f, 1f)}. */
    public static NDArray uniform(PRNGKey key, Shape shape) {
        return uniform(key, shape, 0f, 1f);
    }

    /**
     * Standard-normal samples via Box–Muller, mirroring {@code jax.random.normal}.
     * Each output element consumes two domain-separated uniform draws.
     */
    public static NDArray normal(PRNGKey key, Shape shape) {
        if (key.keyArray() instanceof TracedNDArray traced) {
            Tracer tracer = Tracer.current();
            Var outVar = tracer.nextVar(shape, DType.FLOAT32);
            tracer.addEquation(new Equation(
                List.of(traced.getVar()),
                List.of(outVar),
                Primitive.RANDOM_NORMAL,
                new RandomMeta.Normal(shape)
            ));
            return new TracedNDArray(outVar);
        } else {
            return normalEager(key.keyArray(), shape);
        }
    }

    public static NDArray normalEager(NDArray key, Shape shape) {
        long state = key.toLongArray()[0];
        int n = (int) shape.size();
        float[] data = new float[n];
        if (n >= PARALLEL_INIT_THRESHOLD) {
            java.util.stream.IntStream.range(0, n).parallel().forEach(i -> {
                float u1 = Math.max(rawUniform01(state, 2L * i), 1e-7f);
                float u2 = rawUniform01(state, 2L * i + 1);
                data[i] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
            });
        } else {
            for (int i = 0; i < n; i++) {
                float u1 = Math.max(rawUniform01(state, 2L * i), 1e-7f);
                float u2 = rawUniform01(state, 2L * i + 1);
                data[i] = (float) (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
            }
        }
        return new ConcreteNDArray(data, shape);
    }

    /** Mirrors {@code jax.random.bernoulli(key, p, shape)}: true with probability {@code p}. Returns a real {@code DType.BOOL} array. */
    public static NDArray bernoulli(PRNGKey key, float p, Shape shape) {
        if (key.keyArray() instanceof TracedNDArray traced) {
            Tracer tracer = Tracer.current();
            Var outVar = tracer.nextVar(shape, DType.BOOL);
            tracer.addEquation(new Equation(
                List.of(traced.getVar()),
                List.of(outVar),
                Primitive.RANDOM_BERNOULLI,
                new RandomMeta.Bernoulli(p, shape)
            ));
            return new TracedNDArray(outVar);
        } else {
            return bernoulliEager(key.keyArray(), p, shape);
        }
    }

    public static NDArray bernoulliEager(NDArray key, float p, Shape shape) {
        long state = key.toLongArray()[0];
        int n = (int) shape.size();
        boolean[] data = new boolean[n];
        for (int i = 0; i < n; i++) {
            data[i] = rawUniform01(state, i) < p;
        }
        return new ConcreteNDArray(data, shape);
    }

    /**
     * A uniformly random permutation of {@code 0..n-1} (Fisher–Yates), mirroring
     * {@code jax.random.permutation(key, n)}. Returns a real {@code DType.INT32}
     * array, consistent with how {@code argmax}/{@code argmin} represent indices.
     */
    public static NDArray permutation(PRNGKey key, int n) {
        if (key.keyArray() instanceof TracedNDArray traced) {
            Tracer tracer = Tracer.current();
            Var outVar = tracer.nextVar(new Shape(n), DType.INT32);
            tracer.addEquation(new Equation(
                List.of(traced.getVar()),
                List.of(outVar),
                Primitive.RANDOM_PERMUTATION,
                new RandomMeta.Permutation(n)
            ));
            return new TracedNDArray(outVar);
        } else {
            return permutationEager(key.keyArray(), n);
        }
    }

    public static NDArray permutationEager(NDArray key, int n) {
        long state = key.toLongArray()[0];
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = n - 1; i > 0; i--) {
            float u = rawUniform01(state, i);
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
