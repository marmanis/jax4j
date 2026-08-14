package com.marmanis.jax4j.ml;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a record component as a hyperparameter — {@link Modules} skips it when
 * exposing parameters as a PyTree and preserves it unchanged through
 * flatten/unflatten. Primitive/boxed types, {@code String}s and enums are
 * skipped by default; use {@code @Static} for object-valued hyperparameters
 * (activation function references, config records, etc.) that would otherwise
 * be mistaken for parameter subtrees.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.RECORD_COMPONENT)
public @interface Static {
}
