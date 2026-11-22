# jax4j Modernization & JDK 27 Upgrade — November 2026

Summary of architectural enhancements, modern JAX ergonomics, Project Panama FFM off-heap memory, SPMD named sharding, and JDK 27 upgrade.

## 1. Modern Autodiff Ergonomics & Neural Enhancements (Phase 1)

- **`has_aux` Support**:
  - Added `com.marmanis.jax4j.api.AuxResult<R, A>` record container holding primary value, gradient, and auxiliary payload.
  - Added `JAX.grad(fn, hasAux)` and `JAX.value_and_grad(fn, hasAux)` for both `NDArray` and `PyTree` inputs.
  - Allows functions to return non-differentiated auxiliary data (loss statistics, training state, intermediate activations) alongside the differentiable scalar loss.
- **Fluent Custom VJP (`CustomVjp`)**:
  - Implemented `com.marmanis.jax4j.api.CustomVjp` with fluent builder `CustomVjp.defvjp(fn).fwd(fwdFn).bwd(bwdFn).build()`.
  - Supports stashing arbitrary intermediate residuals during the forward pass and consuming them in the backward pass.
- **Stateful Key Streams (`PRNGSequence`)**:
  - Implemented `com.marmanis.jax4j.api.PRNGSequence.of(seed)` and `PRNGSequence.of(key)`.
  - Provides an ergonomic iterator and key generator stream (`next()`, `take(n)`) eliminating manual `Random.split` boilerplate across iterations.
- **Modern Neural Layers (`RMSNorm`)**:
  - Implemented `com.marmanis.jax4j.ml.nn.RMSNorm` (Root Mean Square Layer Normalization) widely used in modern LLMs (LLaMA, Gemma, Mistral).
- **Verification**:
  - Added comprehensive test suite `Phase1FeaturesTest.java` (7 unit tests).

---

## 2. Memory Architecture & Performance Enhancements (Phase 2)

- **Project Panama FFM Off-Heap Storage (`NativeStorage`)**:
  - Added `ConcreteNDArray.NativeStorage` record backed by `java.lang.foreign.MemorySegment`.
  - Added `NDArray.asMemorySegment()` and `NDArray.isNative()` to the core `NDArray` interface.
  - Added factory methods `ConcreteNDArray.allocateNative(shape, dtype)` and `ConcreteNDArray.fromSegment(segment, shape, dtype)`.
  - Enables zero-copy bidirectional memory exchange with native C/C++ libraries, BLAS/LAPACK, cuBLAS, and external accelerator runtimes.
- **Buffer Donation & In-Place Memory Reuse**:
  - Added `donateInputs` flag to `TornadoJitCompiler`.
  - Allows reusing input device buffers for output allocations during iterative compiled loops, significantly reducing GC and driver allocation overhead.
- **Verification**:
  - Added comprehensive test suite `Phase2FeaturesTest.java` (4 unit tests).

---

## 3. Next-Gen Distributed Compute & Named Sharding (Phase 3)

- **SPMD Distributed Computing**:
  - Implemented `com.marmanis.jax4j.sharding.DeviceMesh`: N-dimensional device grid with named axes (e.g. `data`, `model`).
  - Implemented `com.marmanis.jax4j.sharding.PartitionSpec`: Declarative dimension-to-mesh-axis mapping (e.g. `PartitionSpec.of("data", null)`).
  - Implemented `com.marmanis.jax4j.sharding.NamedSharding`: Declarative association of a `DeviceMesh` and a `PartitionSpec`.
  - Implemented `com.marmanis.jax4j.sharding.ShardMap` and exposed `JAX.shard_map(...)`: Single-Program Multi-Data (SPMD) transformation that partitions inputs, runs local kernels concurrently on device shards, and reassembles global outputs.
- **Verification**:
  - Added comprehensive test suite `Phase3FeaturesTest.java` (5 unit tests).

---

## 4. Upgrade to JDK 27

- Upgraded `jax4j/pom.xml` to `<maven.compiler.release>27</maven.compiler.release>`.
- Configured Maven Surefire plugin with `--enable-native-access=ALL-UNNAMED` for Project Panama Foreign Memory support.
- Fully verified and tested under OpenJDK 27 (Amazon Corretto `27.0.1.1+4-FR`) inside WSL.

---

## Test Suite Evolution

| Milestone | Total Tests | Passing | Skipped |
|---|---|---|---|
| August 2026 (Keras release) | 340 | 324 | 16 |
| November 2026 (Modernization + JDK 27) | **391** | **375** | 16 |
