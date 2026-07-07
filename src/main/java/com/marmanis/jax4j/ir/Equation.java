package com.marmanis.jax4j.ir;

import java.util.List;

/**
 * Represents an equation in a Jaxpr: out = primitive(in1, in2, ...)
 * Supports optional metadata (e.g. for FFI target names).
 */
public record Equation(List<Var> inputs, List<Var> outputs, Primitive primitive, Object metadata) {
    public Equation(List<Var> inputs, List<Var> outputs, Primitive primitive) {
        this(inputs, outputs, primitive, null);
    }

    @Override
    public String toString() {
        String m = metadata != null ? " [" + metadata + "]" : "";
        return String.join(", ", outputs.stream().map(Var::toString).toList()) +
               " = " + primitive + m + " " +
               String.join(", ", inputs.stream().map(Var::toString).toList());
    }
}
