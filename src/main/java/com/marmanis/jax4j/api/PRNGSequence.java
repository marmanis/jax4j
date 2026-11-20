package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.PRNGKey;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * An infinite sequence of independent {@link PRNGKey} instances derived by repeatedly splitting
 * a root key, mirroring common Equinox/JAX pseudo-random generator sequences.
 *
 * <p>While {@code jax4j}'s core functions are purely functional and take explicit keys,
 * initializing deep multi-layer neural networks or stepping through training iterations
 * often requires dozens of keys. {@code PRNGSequence} simplifies this boilerplate without
 * manually indexing into arrays from {@link Random#split(PRNGKey, int)}.
 *
 * <pre>{@code
 * PRNGSequence seq = new PRNGSequence(PRNGKey.key(42));
 * Linear layer1 = Linear.init(seq.next(), 64, 32);
 * Linear layer2 = Linear.init(seq.next(), 32, 10);
 * }</pre>
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class PRNGSequence implements Iterator<PRNGKey>, Iterable<PRNGKey> {

    private PRNGKey currentKey;

    /**
     * Creates a new sequence from an initial root key.
     */
    public PRNGSequence(PRNGKey initialKey) {
        if (initialKey == null) {
            throw new IllegalArgumentException("initialKey must not be null");
        }
        this.currentKey = initialKey;
    }

    /**
     * Creates a new sequence from a seed integer.
     */
    public PRNGSequence(long seed) {
        this(PRNGKey.key(seed));
    }

    /**
     * Returns true, as PRNGKey sequences are infinite.
     */
    @Override
    public boolean hasNext() {
        return true;
    }

    /**
     * Splits the current key into two sub-keys: saves one as the new sequence state,
     * and returns the other for immediate consumption.
     */
    @Override
    public synchronized PRNGKey next() {
        PRNGKey[] pair = Random.split(currentKey, 2);
        this.currentKey = pair[0];
        return pair[1];
    }

    /**
     * Generates an array of {@code count} independent keys from the sequence.
     */
    public synchronized PRNGKey[] next(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative: " + count);
        }
        PRNGKey[] result = new PRNGKey[count];
        for (int i = 0; i < count; i++) {
            result[i] = next();
        }
        return result;
    }

    @Override
    public Iterator<PRNGKey> iterator() {
        return this;
    }
}
