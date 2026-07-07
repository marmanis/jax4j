package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.tracing.Tracer;
import com.marmanis.jax4j.tracing.TracedNDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Foreign Function Interface for jax4j.
 */
public class FFI {
    private static final Map<String, FFITarget> targets = new HashMap<>();

    public record FFITarget(
        String name, 
        Function<NDArray[], NDArray> implementation,
        BiFunction<NDArray, NDArray[], NDArray[]> vjp // (grad_out, inputs) -> grad_inputs
    ) {}

    /**
     * Registers a foreign function target with an optional VJP.
     */
    public static void register(String name, Function<NDArray[], NDArray> implementation, 
                                BiFunction<NDArray, NDArray[], NDArray[]> vjp) {
        targets.put(name, new FFITarget(name, implementation, vjp));
    }

    public static void register(String name, Function<NDArray[], NDArray> implementation) {
        register(name, implementation, null);
    }

    /**
     * Calls a registered foreign function.
     */
    public static NDArray call(String name, Shape outShape, NDArray... inputs) {
        if (Tracer.current() != null) {
            // During tracing, record as an FFI call
            // We'll store the target name as metadata (simplified for MVP)
            return ((TracedNDArray) inputs[0]).applyFFI(name, outShape, inputs);
        }
        
        // Eager execution
        FFITarget target = targets.get(name);
        if (target == null) throw new IllegalArgumentException("Unknown FFI target: " + name);
        return target.implementation().apply(inputs);
    }

    public static FFITarget getTarget(String name) {
        return targets.get(name);
    }
}
