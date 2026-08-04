package com.marmanis.jax4j.ir;

import com.marmanis.jax4j.core.NDArray;

import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Metadata for the CUSTOM_VJP primitive: the forward function and the
 * user-supplied VJP function.
 *
 * <p>{@code fn} computes the forward pass: {@code output = fn(input)}.
 * {@code vjpFn} computes the gradient: {@code grad_input = vjpFn(input, grad_output)}.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public record CustomVjpMeta(
        Function<NDArray, NDArray> fn,
        BiFunction<NDArray, NDArray, NDArray> vjpFn) {}
