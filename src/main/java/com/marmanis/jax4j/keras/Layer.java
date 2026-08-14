package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Base class for all Keras-style layers. A layer is a mutable holder for its
 * own weights (created lazily by {@link #build(Shape, PRNGKey)} once the input
 * shape is known) plus a pure {@link #forward} operator over those weights.
 *
 * <p>Training runs functionally: {@link Model} freezes each layer's weights
 * into a {@link com.marmanis.jax4j.pytree.PyTree} and passes overrides through
 * {@link #call(NDArray, Map, boolean)}, so the same {@link #forward} is used
 * both for eager prediction and for gradient-tape execution.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public abstract class Layer {

    private static final Map<String, AtomicInteger> NAME_COUNTERS = new ConcurrentHashMap<>();

    protected final String name;
    protected boolean built = false;
    protected Shape inputShape;
    protected final List<String> paramNames = new ArrayList<>();
    protected final Map<String, NDArray> params = new LinkedHashMap<>();

    protected Layer() { this(null); }

    protected Layer(String name) {
        this.name = (name != null) ? name : autoName();
    }

    private String autoName() {
        String base = defaultNamePrefix();
        AtomicInteger ctr = NAME_COUNTERS.computeIfAbsent(base, _ -> new AtomicInteger());
        int i = ctr.getAndIncrement();
        return (i == 0) ? base : base + "_" + i;
    }

    /**
     * Snake-case base name used when auto-generating this layer's name. Subclasses
     * override to match the Keras convention (e.g. {@code Dense} -> {@code "dense"}).
     */
    protected String defaultNamePrefix() {
        String s = getClass().getSimpleName();
        StringBuilder out = new StringBuilder(s.length() + 4);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isUpperCase(c)) {
                if (i > 0) out.append('_');
                out.append(Character.toLowerCase(c));
            } else out.append(c);
        }
        return out.toString();
    }

    /** Compute the output shape given the input shape, without building. */
    public abstract Shape computeOutputShape(Shape inputShape);

    /**
     * Lazily creates the layer's parameters from a fresh PRNG key. Subclasses
     * fill {@link #params} (registering keys in {@link #paramNames}). Called
     * exactly once — repeated calls with the same input shape are no-ops.
     */
    public final void build(Shape inputShape, PRNGKey key) {
        if (built) return;
        this.inputShape = inputShape;
        doBuild(inputShape, key);
        built = true;
    }

    /** Subclass hook for {@link #build}: populate {@link #params}/{@link #paramNames}. */
    protected abstract void doBuild(Shape inputShape, PRNGKey key);

    /**
     * The pure functional forward pass, called with resolved parameters.
     * @param inputs         one entry per upstream tensor
     * @param effectiveParams the layer's parameters (unmodified when {@code paramOverrides}
     *                        was {@code null}, or the values pulled out of the override map)
     * @param training        {@code true} during {@code fit}, {@code false} during {@code predict}/{@code evaluate}
     */
    protected abstract NDArray forward(List<NDArray> inputs, Map<String, NDArray> effectiveParams, boolean training);

    /** Convenience single-input entry point. */
    public final NDArray call(NDArray input, Map<String, NDArray> paramOverrides, boolean training) {
        return call(List.of(input), paramOverrides, training);
    }

    public final NDArray call(List<NDArray> inputs, Map<String, NDArray> paramOverrides, boolean training) {
        Map<String, NDArray> effective = resolveParams(paramOverrides);
        return forward(inputs, effective, training);
    }

    /** Eager forward pass using this layer's own params (no gradient tape). */
    public final NDArray call(NDArray input) {
        return call(input, null, false);
    }

    private Map<String, NDArray> resolveParams(Map<String, NDArray> overrides) {
        if (overrides == null) return params;
        if (paramNames.isEmpty()) return Collections.emptyMap();
        Map<String, NDArray> resolved = new LinkedHashMap<>();
        for (String p : paramNames) {
            NDArray v = overrides.get(name + "/" + p);
            resolved.put(p, v != null ? v : params.get(p));
        }
        return resolved;
    }

    /**
     * Functional-API entry point: records an edge in the graph and returns a
     * new symbolic tensor. Does not build the layer — {@link Model} builds all
     * layers in topo order once its {@code inputs}/{@code outputs} are known.
     */
    public KerasTensor apply(KerasTensor input) {
        return new KerasTensor(computeOutputShape(input.shape()), input.dtype(),
            new KerasTensor.LayerNode(this, List.of(input)));
    }

    /**
     * Multi-input variant. Subclasses that consume multiple upstream tensors
     * override {@link #computeOutputShape(List)} instead of the single-input version.
     */
    public KerasTensor apply(KerasTensor... inputs) {
        List<KerasTensor> list = List.of(inputs);
        return new KerasTensor(computeOutputShape(list), inputs[0].dtype(),
            new KerasTensor.LayerNode(this, list));
    }

    /** Multi-input shape rule. Default: delegates to the single-input version on {@code inputs[0]}. */
    protected Shape computeOutputShape(List<KerasTensor> inputs) {
        return computeOutputShape(inputs.get(0).shape());
    }

    public String getName() { return name; }
    public boolean isBuilt() { return built; }
    public Map<String, NDArray> getParams() { return Collections.unmodifiableMap(params); }
    public List<String> getParamNames() { return Collections.unmodifiableList(paramNames); }

    public int countParams() {
        int total = 0;
        for (NDArray p : params.values()) total += (int) p.shape().size();
        return total;
    }

    /** Test hook: reset the per-subclass name counters (used by tests). */
    public static void resetNameCounters() {
        NAME_COUNTERS.clear();
    }
}
