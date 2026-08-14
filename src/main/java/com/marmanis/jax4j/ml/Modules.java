package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Reflection-based bridge between record-shaped {@link Module}s and jax4j's
 * {@link PyTree} data model. Enables users to write models as immutable
 * records and still use {@code grad}/{@code jit}/optimizer updates that need
 * a flat, structure-preserving view of every parameter.
 *
 * <p>Traversal rules for record components:
 * <ul>
 *   <li>{@link Static}-annotated → skipped (kept as hyperparameter)</li>
 *   <li>{@link NDArray} → parameter leaf</li>
 *   <li>{@link Module} → recurse (nested subtree)</li>
 *   <li>{@link List} → ordered subtree; element type decides leaf vs. recurse</li>
 *   <li>{@link Module}[] → ordered subtree of nested modules</li>
 *   <li>primitives / boxed numbers / {@code String} / enums → skipped</li>
 * </ul>
 *
 * <p>Canonical record constructors are looked up reflectively and cached per class.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Modules {
    private Modules() {}

    private static final ConcurrentHashMap<Class<?>, Constructor<?>> CTOR_CACHE = new ConcurrentHashMap<>();

    // ---- flatten / unflatten -------------------------------------------------

    /**
     * Recursive record walk producing a {@link PyTree.MapNode} whose keys are
     * record component names (in declaration order). Static/skipped components
     * are omitted from the map.
     */
    public static PyTree flatten(Module m) {
        return flatten(m, x -> true);
    }

    /**
     * Recursive record walk producing a {@link PyTree.MapNode} where parameters
     * matching the filter are kept as leaves, and non-matching parameters are
     * replaced by null leaves.
     */
    public static PyTree flatten(Module m, java.util.function.Predicate<NDArray> filter) {
        Map<String, PyTree> out = new LinkedHashMap<>();
        for (RecordComponent rc : m.getClass().getRecordComponents()) {
            if (isSkipped(rc)) continue;
            Object value = invoke(rc, m);
            PyTree sub = valueToTreeFiltered(value, filter);
            if (sub != null) out.put(rc.getName(), sub);
        }
        return new PyTree.MapNode(out);
    }

    /**
     * Rebuilds a module of the same class as {@code template}, substituting
     * parameter leaves from {@code tree}. Static components are copied from
     * {@code template}. The tree must have the same shape {@link #flatten}
     * would have produced from {@code template}.
     */
    @SuppressWarnings("unchecked")
    public static <M extends Module> M unflatten(M template, PyTree tree) {
        if (!(tree instanceof PyTree.MapNode(var children))) {
            throw new IllegalArgumentException("Modules.unflatten expects a MapNode at each module level, got " + tree.getClass().getSimpleName());
        }
        RecordComponent[] components = template.getClass().getRecordComponents();
        Object[] args = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            RecordComponent rc = components[i];
            Object templateValue = invoke(rc, template);
            if (isSkipped(rc)) {
                args[i] = templateValue;
            } else {
                PyTree sub = children.get(rc.getName());
                if (sub == null) {
                    args[i] = templateValue;
                } else {
                    args[i] = treeToValue(templateValue, sub);
                }
            }
        }
        return (M) construct(template.getClass(), components, args);
    }

    // ---- tree maps -----------------------------------------------------------

    /** Apply {@code f} to every parameter leaf, preserving structure. */
    public static <M extends Module> M treeMap(M m, Function<NDArray, NDArray> f) {
        PyTree tree = flatten(m);
        PyTree mapped = PyTrees.map(f, tree);
        return unflatten(m, mapped);
    }

    /** Zip {@code a} and {@code b} leafwise through {@code f} (structures must match). */
    public static <M extends Module> M treeMap2(M a, M b, BiFunction<NDArray, NDArray, NDArray> f) {
        PyTree ta = flatten(a);
        PyTree tb = flatten(b);
        PyTree combined = PyTrees.map2(f, ta, tb);
        return unflatten(a, combined);
    }

    /** Zero-array copy: every leaf replaced by a same-shape/dtype zero array. */
    public static <M extends Module> M zerosLike(M m) {
        return treeMap(m, leaf -> zerosArray(leaf.shape(), leaf.dtype()));
    }

    // ---- filter helpers over grad / jit --------------------------------------

    /**
     * Binds {@code batch} into a {@code (module, batch) -> loss} function to
     * produce the {@code module -> loss} shape {@link #filterGrad} consumes.
     */
    public static <M extends Module> Function<M, NDArray> filterLoss(
            BiFunction<M, NDArray, NDArray> lossFn, NDArray batch) {
        return module -> lossFn.apply(module, batch);
    }

    /**
     * Returns a module of the same shape as {@code module} whose leaves are
     * the gradients of {@code lossFn} w.r.t. each parameter, via
     * {@link Grad#gradTree(Function)} over the module's flattened pytree.
     */
    public static <M extends Module> M filterGrad(Function<M, NDArray> lossFn, M module) {
        return filterGrad(lossFn, module, x -> true);
    }

    /**
     * Differentiates w.r.t the filtered parameters, returning a gradient module
     * with null gradients for any non-matching (frozen) parameters.
     */
    public static <M extends Module> M filterGrad(Function<M, NDArray> lossFn, M module, java.util.function.Predicate<NDArray> filter) {
        PyTree paramsTree = flatten(module, filter);
        Function<PyTree, NDArray> treeFn = params -> {
            M paramsMod = unflatten(module, params);
            M combinedMod = combine(paramsMod, module);
            return lossFn.apply(combinedMod);
        };
        PyTree gradTree = Grad.gradTree(treeFn).apply(paramsTree);
        return unflatten(module, gradTree);
    }

    /**
     * Binds {@code module}'s parameters as constants and JIT-compiles
     * {@code x -> fn(module, x)} for repeated evaluation with the same
     * module (typical inference use). The returned function is
     * {@link JAX#jit}-cached on the input's shape/dtype signature.
     */
    public static <M extends Module> Function<NDArray, NDArray> filterJit(
            M module, BiFunction<M, NDArray, NDArray> fn) {
        Function<NDArray, NDArray> bound = x -> fn.apply(module, x);
        return JAX.jit(bound);
    }

    // ---- internals -----------------------------------------------------------

    /**
     * Decide whether a component contributes parameters. Static components,
     * primitive/boxed numeric types, strings, and enums are always skipped.
     */
    private static boolean isSkipped(RecordComponent rc) {
        if (rc.getAnnotation(Static.class) != null) return true;
        Class<?> t = rc.getType();
        if (t.isPrimitive()) return true;
        if (t == Boolean.class || t == Byte.class || t == Short.class
                || t == Integer.class || t == Long.class || t == Float.class
                || t == Double.class || t == Character.class) return true;
        if (t == String.class) return true;
        if (t.isEnum()) return true;
        return false;
    }

    private static Object invoke(RecordComponent rc, Object record) {
        try {
            return rc.getAccessor().invoke(record);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("Failed to read component " + rc.getName(), e);
        }
    }

    /**
     * Convert a live component value into a subtree. Returns {@code null} for
     * unrecognized values so the caller can drop the key.
     */
    private static PyTree valueToTree(Object value) {
        return valueToTreeFiltered(value, x -> true);
    }

    private static PyTree valueToTreeFiltered(Object value, java.util.function.Predicate<NDArray> filter) {
        if (value == null) return null;
        if (value instanceof NDArray arr) return new PyTree.Leaf(filter.test(arr) ? arr : null);
        if (value instanceof Module sub) return flatten(sub, filter);
        if (value instanceof List<?> list) return listToTreeFiltered(list, filter);
        if (value instanceof Module[] arr) {
            List<PyTree> children = new ArrayList<>(arr.length);
            for (Module m : arr) children.add(flatten(m, filter));
            return new PyTree.ListNode(children);
        }
        return null;
    }

    private static PyTree listToTree(List<?> list) {
        return listToTreeFiltered(list, x -> true);
    }

    private static PyTree listToTreeFiltered(List<?> list, java.util.function.Predicate<NDArray> filter) {
        List<PyTree> children = new ArrayList<>(list.size());
        for (Object el : list) {
            if (el instanceof NDArray arr) children.add(new PyTree.Leaf(filter.test(arr) ? arr : null));
            else if (el instanceof Module sub) children.add(flatten(sub, filter));
            else children.add(new PyTree.Leaf(null));
        }
        return new PyTree.ListNode(children);
    }

    /** Reconstruct a component value from a template shape and a subtree. */
    private static Object treeToValue(Object templateValue, PyTree tree) {
        if (templateValue instanceof NDArray && tree instanceof PyTree.Leaf(var value)) {
            return value;
        }
        if (templateValue instanceof Module sub) {
            return unflatten(sub, tree);
        }
        if (templateValue instanceof List<?> list) {
            return listFromTree(list, tree);
        }
        if (templateValue instanceof Module[] arr) {
            List<PyTree> children = ((PyTree.ListNode) tree).children();
            Module[] out = arr.clone();
            for (int i = 0; i < arr.length; i++) out[i] = unflatten(arr[i], children.get(i));
            return out;
        }
        return templateValue;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static List listFromTree(List<?> templateList, PyTree tree) {
        List<PyTree> children = ((PyTree.ListNode) tree).children();
        if (children.size() != templateList.size()) {
            throw new IllegalArgumentException("List length mismatch on unflatten: "
                + children.size() + " vs template " + templateList.size());
        }
        List out = new ArrayList(children.size());
        for (int i = 0; i < children.size(); i++) {
            Object templateEl = templateList.get(i);
            if (templateEl instanceof NDArray) {
                out.add(((PyTree.Leaf) children.get(i)).value());
            } else if (templateEl instanceof Module m) {
                out.add(unflatten(m, children.get(i)));
            } else {
                out.add(templateEl);
            }
        }
        return out;
    }

    /**
     * Combines a parameter module (which may contain null fields for frozen parameters)
     * with a template module (which contains original values), falling back to
     * template's values for any null leaves.
     */
    public static <M extends Module> M combine(M params, M template) {
        PyTree pTree = flatten(params, x -> true);
        return unflattenCombine(template, pTree);
    }

    @SuppressWarnings("unchecked")
    private static <M extends Module> M unflattenCombine(M template, PyTree tree) {
        if (!(tree instanceof PyTree.MapNode(var children))) {
            throw new IllegalArgumentException("Modules.combine expects a MapNode at each module level, got " + tree.getClass().getSimpleName());
        }
        RecordComponent[] components = template.getClass().getRecordComponents();
        Object[] args = new Object[components.length];
        for (int i = 0; i < components.length; i++) {
            RecordComponent rc = components[i];
            Object templateValue = invoke(rc, template);
            if (isSkipped(rc)) {
                args[i] = templateValue;
            } else {
                PyTree sub = children.get(rc.getName());
                if (sub == null) {
                    args[i] = templateValue;
                } else {
                    args[i] = treeToValueCombine(templateValue, sub);
                }
            }
        }
        return (M) construct(template.getClass(), components, args);
    }

    private static Object treeToValueCombine(Object templateValue, PyTree tree) {
        if (templateValue instanceof NDArray && tree instanceof PyTree.Leaf(var value)) {
            return value != null ? value : templateValue;
        }
        if (templateValue instanceof Module sub) {
            return unflattenCombine(sub, tree);
        }
        if (templateValue instanceof List<?> list) {
            return listFromTreeCombine(list, tree);
        }
        if (templateValue instanceof Module[] arr) {
            List<PyTree> children = ((PyTree.ListNode) tree).children();
            Module[] out = arr.clone();
            for (int i = 0; i < arr.length; i++) out[i] = unflattenCombine(arr[i], children.get(i));
            return out;
        }
        return templateValue;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static List listFromTreeCombine(List<?> templateList, PyTree tree) {
        List<PyTree> children = ((PyTree.ListNode) tree).children();
        if (children.size() != templateList.size()) {
            throw new IllegalArgumentException("List length mismatch on unflatten: "
                + children.size() + " vs template " + templateList.size());
        }
        List out = new ArrayList(children.size());
        for (int i = 0; i < children.size(); i++) {
            Object templateEl = templateList.get(i);
            if (templateEl instanceof NDArray) {
                NDArray value = ((PyTree.Leaf) children.get(i)).value();
                out.add(value != null ? value : templateEl);
            } else if (templateEl instanceof Module m) {
                out.add(unflattenCombine(m, children.get(i)));
            } else {
                out.add(templateEl);
            }
        }
        return out;
    }

    private static Object construct(Class<?> cls, RecordComponent[] components, Object[] args) {
        Constructor<?> ctor = CTOR_CACHE.computeIfAbsent(cls, c -> {
            Class<?>[] paramTypes = new Class<?>[components.length];
            for (int i = 0; i < components.length; i++) paramTypes[i] = components[i].getType();
            try {
                Constructor<?> k = c.getDeclaredConstructor(paramTypes);
                k.setAccessible(true);
                return k;
            } catch (NoSuchMethodException e) {
                throw new RuntimeException("No canonical constructor for " + c.getName(), e);
            }
        });
        try {
            return ctor.newInstance(args);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("Failed to construct " + cls.getName(), e);
        }
    }

    private static NDArray zerosArray(Shape shape, DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> new ConcreteNDArray(new float[(int) shape.size()], shape);
            case FLOAT64 -> new ConcreteNDArray(new double[(int) shape.size()], shape);
            case INT32 -> new ConcreteNDArray(new int[(int) shape.size()], shape);
            case INT64 -> new ConcreteNDArray(new long[(int) shape.size()], shape);
            case BOOL -> new ConcreteNDArray(new boolean[(int) shape.size()], shape);
        };
    }
}
