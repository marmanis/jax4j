package com.marmanis.jax4j.ir;

import com.marmanis.jax4j.core.NDArray;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A JAX expression (Jaxpr) representing a traced computation.
 */
public record Jaxpr(List<Var> inVars, List<Var> outVars, List<Equation> equations, Map<Integer, NDArray> consts) {
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{ lambda ");
        sb.append(inVars.stream().map(Var::toString).collect(Collectors.joining(", ")));
        sb.append(" ; ");
        if (!consts.isEmpty()) {
            sb.append("consts: ").append(consts.keySet()).append(" ; ");
        }
        sb.append("\n  let\n");
        for (Equation eq : equations) {
            sb.append("    ").append(eq).append("\n");
        }
        sb.append("  in (");
        sb.append(outVars.stream().map(Var::toString).collect(Collectors.joining(", ")));
        sb.append(") }");
        return sb.toString();
    }
}
