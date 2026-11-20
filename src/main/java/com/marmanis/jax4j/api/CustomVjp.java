package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.NDArray;

import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Fluent builder for custom vector-Jacobian products (VJP), mirroring {@code jax.custom_vjp}.
 *
 * <p>Allows defining a custom backward rule (gradient calculation) for a function, overriding
 * automatic differentiation. Useful for numerically stabilizing functions (e.g. log-sum-exp,
 * custom activation functions, or gradient clipping).
 *
 * <pre>{@code
 * Function<NDArray, NDArray> stableFn = CustomVjp.builder(x -> rawForward(x))
 *     .defvjp((x, g) -> customGradient(x, g))
 *     .build();
 * }</pre>
 *
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class CustomVjp {

    private CustomVjp() {}

    /**
     * Starts building a custom VJP around the given forward function.
     *
     * @param fn the forward function
     * @return a builder instance
     */
    public static Builder builder(Function<NDArray, NDArray> fn) {
        return new Builder(fn);
    }

    /**
     * Direct convenience definition of a custom VJP function.
     *
     * @param fn    the forward function: {@code y = fn(x)}
     * @param vjpFn the VJP function: {@code grad_x = vjpFn(x, grad_y)}
     * @return a differentiable function using the custom backward rule
     */
    public static Function<NDArray, NDArray> defvjp(
            Function<NDArray, NDArray> fn,
            BiFunction<NDArray, NDArray, NDArray> vjpFn) {
        return JAX.customVjp(fn, vjpFn);
    }

    public static final class Builder {
        private final Function<NDArray, NDArray> forwardFn;
        private BiFunction<NDArray, NDArray, NDArray> vjpFn;

        private Builder(Function<NDArray, NDArray> forwardFn) {
            this.forwardFn = forwardFn;
        }

        /**
         * Defines the custom vector-Jacobian product.
         *
         * @param vjpFn receives {@code (input, gradOutput)} and returns {@code gradInput}
         * @return this builder
         */
        public Builder defvjp(BiFunction<NDArray, NDArray, NDArray> vjpFn) {
            this.vjpFn = vjpFn;
            return this;
        }

        /**
         * Builds the differentiable function.
         */
        public Function<NDArray, NDArray> build() {
            if (vjpFn == null) {
                throw new IllegalStateException("vjpFn must be defined before calling build()");
            }
            return JAX.customVjp(forwardFn, vjpFn);
        }
    }
}
