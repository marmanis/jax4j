# jax4j - JAX for Java with TornadoVM

`jax4j` is a high-performance numerical computing library for Java, inspired by Google's [JAX](https://github.com/jax-ml/jax). It brings functional transformations like Just-In-Time (JIT) compilation, automatic differentiation (`grad`), and vectorization (`vmap`) to the Java ecosystem, leveraging [TornadoVM](https://github.com/beehive-lab/TornadoVM) for hardware acceleration across GPUs and other heterogeneous devices.

## Features

- **Functional IR (Jaxpr)**: Computations are traced into a simplified intermediate representation for transformation and optimization.
- **Automatic Differentiation**: Reverse-mode AD via `grad`, including multi-argument gradients (`argnums`), gradients over nested parameter pytrees, and higher-order gradients (`grad(grad(f))`).
- **Data-parallel execution (`pmap`)**: `JAX.pmap(fn, devices)` splits the input's leading axis across a list of devices (one shard per device), runs `fn` on each shard concurrently, and stacks the results back — mirroring `jax.pmap` with `in_axes=0, out_axes=0`. Collective ops inside the body: `Lax.psum(x)` all-reduces (sums) across devices; `Lax.allGather(x, numDevices)` gathers all shards and returns the full stacked array to every device. Both collectives use a two-phase `CyclicBarrier` (deposit → read → release) so that sequential collectives within the same body never race. Fully compose with `grad`: `JAX.grad(x -> JAX.pmap(f, devices).apply(x).sum())` differentiates through the pmap via a `PMAP` primitive whose VJP runs the backward pass on each shard concurrently — and the VJP of `psum` is itself a `psum` (the correct linear-map transpose: each replica's input x_i contributed to every other replica's psum output, so the incoming gradient must also be all-reduced). Without real hardware, pass `List.of(new Device(null), new Device(null))` for two host-backed synthetic shards — all sharding and collective logic is exercised identically on the CPU. Also composes with `vmap`: `vmap(x -> pmap(f, D)(x))` runs pmap independently per batch element.
- **Automatic Batching (`vmap`)**: Vectorizes a function written for a single example over a leading batch dimension, with per-primitive batching rules (elementwise ops, reductions, and matrix multiply).
- **Pytrees**: Nested `List`/`Map<String, _>` containers of `NDArray` leaves that `grad` and other transforms can flatten, map, and rebuild — for working with structured parameter sets (e.g. `{"w": ..., "b": ...}`) instead of single flat arrays.
- **Heterogeneous Hardware Acceleration**: Powered by TornadoVM, supporting OpenCL, CUDA (PTX), SPIR-V, and Metal backends — including OpenCL targeting the multi-core CPU itself (e.g. via POCL), not just discrete GPUs.
- **Explicit Multi-Device Placement**: `NDArray.to(Device)` places an array on a specific device (mirroring `jax.device_put`); elementwise ops on arrays explicitly placed on the same non-host device dispatch to that device via TornadoVM (`TornadoExecutionPlan.withDevice(...)`), with automatic fallback to the host if the TornadoVM runtime isn't available. `MultiDevice.map` runs independent work across several devices concurrently.
- **Activations**: `tanh`/`relu`/`sigmoid` are core, differentiable, device-dispatchable `NDArray` primitives (mirroring `jax.numpy.tanh` etc.); `com.marmanis.jax4j.api.Nn#softmax` is a row-wise softmax over the last axis, built from `exp`/`sum(axis, keepDims)`/`div` (mirroring `jax.nn.softmax`).
- **Axis-aware reductions**: `NDArray.sum(axis, keepDims)`/`mean(axis, keepDims)` reduce along a single axis (negative axes count from the end, e.g. `-1` for the last dimension), mirroring `numpy.sum(axis=, keepdims=)`. Fully differentiable (`SUM_AXIS`/`MEAN_AXIS` VJP rules) and `vmap`-composable (the batch axis shifts the reduction axis by one automatically). The full-reduction `sum()`/`mean()` (collapsing to a scalar) remain available unchanged.
- **Selection and comparison ops**: `NDArray.gt`/`ge`/`lt`/`le`/`eq`/`ne` are elementwise, broadcasting comparisons (mirroring `jax.lax.gt` etc.) returning a real `DType.BOOL` array; `max`/`min` are elementwise and fully differentiable (gradient flows to whichever operand "won", ties going to the first argument); `clip(lo, hi)` is `this.max(lo).min(hi)`; `argmax(axis)`/`argmin(axis)` return the index of the extremum along an axis as a real `DType.INT32` array. Comparisons and `argmax`/`argmin` carry a zero gradient (the standard AD convention) rather than throwing when used inside a differentiated function, so `com.marmanis.jax4j.api.Numpy#where(cond, x, y)` — composed entirely from `mul`/`add`/`sub`, casting a BOOL `cond` to FLOAT32 internally — is a real, differentiable data-dependent branch: `where(x.gt(0), x*x, -x)` differentiates correctly through whichever side `cond` selects. All ten ops are `vmap`-composable (`argmax`/`argmin`'s axis shifts by one when batched, same as `sum`/`mean`).
- **Real INT32/BOOL/FLOAT64/INT64 dtypes**: jax4j has genuinely typed storage for `FLOAT32`/`INT32`/`BOOL`/`FLOAT64`/`INT64` — not a label on `float[]` data. `NDArray.astype(DType)` is the principled, explicit way to convert between them (any FLOAT→INT cast truncates toward zero; any numeric type→BOOL is "nonzero"; BOOL→numeric is 1/0; INT64→INT32 narrows, same overflow caveat as Java's own narrowing cast), traceable via a `CAST` primitive whose VJP is identity for a true same-floating-dtype no-op cast (FLOAT32→FLOAT32 or FLOAT64→FLOAT64) and zero for every other dtype pair. **Arithmetic** (`add`/`sub`/`mul`/`div`/`max`/`min`/unary math/`dot`/reductions/`argmax`/`argmin`) requires both operands to share the same *floating* dtype — `FLOAT32` or `FLOAT64`, no implicit mixing — and throws a clear "use `.astype()`" error otherwise; `FLOAT64` arithmetic always runs on the host (no TornadoVM kernel), unlike `FLOAT32` which dispatches to a device when explicitly placed. **`INT32`/`INT64`** mirror each other (storage/`astype`/comparisons/`argmax` output only, no arithmetic — exactly like `INT32` never got arithmetic, `INT64` doesn't either); `INT64` comparisons compare as genuine `long` (not cast through `double`, which would lose precision beyond 2^53). Comparisons accept `FLOAT32`-vs-`FLOAT32`, `INT32`-vs-`INT32`, `FLOAT64`-vs-`FLOAT64`, or `INT64`-vs-`INT64` (same dtype both sides), always returning `BOOL`. `com.marmanis.jax4j.api.Numpy#take(table, indices)` is an embedding-table lookup (`table[vocab, dim]` FLOAT32 + `indices[...]` INT32 of any shape → `output[...indices.shape, dim]` FLOAT32), differentiable w.r.t. `table` via a scatter-add VJP (the standard embedding backward, correctly accumulating when an index repeats) and `vmap`-composable for free since the eager gather loop already generalizes over indices' shape — stays FLOAT32/INT32-only, not extended to FLOAT64/INT64. `Random.bernoulli`/`permutation` return real BOOL/INT32 arrays (also not extended). Comparisons/`argmax`/casts/gather are host-only (no TornadoVM kernels), consistent with the existing `SUM`/`MEAN`/`DOT` reduction precedent.
- **Explicit-key randomness (`com.marmanis.jax4j.api.Random`)**: Mirrors `jax.random`'s explicit-key design — no hidden global RNG state; every sampling call (`uniform`/`normal`/`bernoulli`/`permutation`) is a pure function of a `PRNGKey` and a shape, and `Random.split(key)` derives independent child keys for unrelated uses (e.g. one subkey per layer's init, another for a dropout mask). The generator is a SplitMix64-style 64-bit mix (splittable, reproducible, statistically sound) rather than a port of JAX's Threefry — not bit-compatible with Python JAX, but gets the same "no implicit state, fully reproducible" guarantees. `glorotUniform`/`heNormal` are composed convenience initializers. Because `PRNGKey` is a plain Java value (never an `NDArray`, never traced), sampling always runs eagerly and the result is captured as an ordinary constant when used inside a traced function — no new primitive, VJP rule, or `vmap` rule was needed. `com.marmanis.jax4j.api.Nn#dropout(x, key, rate, training)` is built the same way (a `Random.bernoulli` mask combined via `mul`/`div`), so it differentiates correctly for free (zero gradient at dropped positions, `1/(1-rate)` at kept positions). For per-example randomness in a batch, sample directly at the batch-inclusive shape rather than vmapping over an array of keys.
- **Control flow**: `com.marmanis.jax4j.api.Lax` mirrors `jax.lax`'s `cond`/`while_loop`/`scan`/`fori_loop`. Unlike a plain Java `if`/`for` wrapped around traced ops, `cond` and `scan` record themselves as a single equation in the surrounding Jaxpr (with the branch/step functions traced into their own nested Jaxprs), so they're real, differentiable parts of the computation graph under `grad`/`gradTree` rather than being unrolled away outside it — `scan`'s backward pass is itself an internal reverse scan over the step function, mirroring how JAX differentiates `lax.scan`. `while_loop` has no VJP and is not reverse-mode differentiable (its iteration count is data-dependent), exactly like upstream `jax.lax.while_loop`. `fori_loop`'s bound is a plain Java `int`, so it's simply unrolled — already fully traceable with no new primitive needed.
- **FFT / DCT-I (`com.marmanis.jax4j.api.Fft`)**: Iterative radix-2 Cooley-Tukey `fft`/`ifft` on complex FLOAT32 or FLOAT64 inputs (length a power of two), plus `dctI` — the real transform Chebyshev value/coefficient conversions use — built on top of the FFT via the length-`2(N-1)` mirror trick. Host-only for now (no TornadoVM kernel, no VJP/vmap rules; treated as a numerical primitive alongside `CAST`/gather). Powers the sibling [chebfun4j](../chebfun4j) project's values ↔ coefficients transform.
- **Trace caching (`JAX.jit` / `JAX.jitGrad` / `JAX.jitGradTree`)**: `JAX.jit(fn)` wraps a function so it is traced once per distinct `(shape, dtype)` input signature and executed via a Jaxpr interpreter on subsequent calls — the tracing cost is paid on the first call at each new signature, later calls skip straight to `Grad.forwardInterpret`. `JAX.jitGrad(fn)` and `JAX.jitGradTree(fn)` do the same for reverse-mode AD: they cache the *traced* Jaxpr and reuse it, running `Grad.backwardInterpret` per call. Both are transparent under nested tracing (a jitted call inside `JAX.grad` falls through eagerly so the outer trace records ops correctly). `Grad.forwardInterpret` and `Grad.backwardInterpret` are also exposed publicly for consumers who want to trace once manually and interpret many times (see chebfun4j's `NonlinearChebop.AutodiffBatched`).
- **Java 24+ Optimized**: Built with modern Java features like records and improved JIT for maximum efficiency.
- **Seamless Integration**: Familiar NumPy-like API for NDArray operations.

Not yet implemented:
- `pmap` with multi-axis or non-zero `in_axes`/`out_axes` — the current implementation always splits axis 0 and stacks axis 0. Structured pytree inputs/outputs (e.g. `pmap` over a dict of parameters) are not yet supported.
- Multi-axis reductions (`axis=(0, 1)` tuples) — only a single axis at a time.
- `Lax.cond`/`whileLoop`'s `pred`/`condFn` is still consumed as a *concrete* value at the point the equation is interpreted (it's read with `.toFloatArray()` to pick a branch/decide whether to keep looping), so a data-dependent predicate built from the new comparison ops (e.g. `x.gt(0)`) works fine as `cond`'s *operand* but can't itself be threaded through `cond`/`whileLoop` symbolically the way `Numpy.where` can — there's no lazy/traced branch selection, only eager dispatch by a resolved value.
- TornadoVM kernels for `SUM`/`MEAN`/`SUM_AXIS`/`MEAN_AXIS` (matrix multiply (`DOT`) and all elementwise ops — `ADD/SUB/MUL/DIV/EXP/LOG/SIN/COS/TANH/RELU/SIGMOID` — dispatch to TornadoVM now; reductions still need TornadoVM's `@Reduce`-style annotations, so they always run on the host regardless of device placement). `Lax.cond`/`whileLoop`/`scan` are interpreted on the host only.
- A real Jaxpr→TornadoVM compiler for `jit` (it currently traces and falls back to eager execution; only *primitive*-level execution is device-aware so far).
- Device-aware broadcasting: combining differently-shaped arrays on a non-host device still computes correctly, but always via the host loop (the device tag is preserved on the result either way).
- `vmap` has no batching rules for `Lax.cond`/`whileLoop`/`scan` yet.
- Forward-mode AD (`jvp`).
- A general dtype-promotion lattice — arithmetic ops require both operands to share the exact same dtype and throw rather than implicitly promoting/coercing (BOOL/INT32/INT64 inputs, or even FLOAT32-vs-FLOAT64 mixing); `.astype()` is always explicit. Beyond `FLOAT32`/`INT32`/`BOOL`/`FLOAT64`/`INT64`, no other dtypes (e.g. complex, float16/bf16) are planned.
- General multi-axis fancy indexing/slicing (random crop, random flip, arbitrary-axis gather) — `Numpy.take` covers the single embedding-table-lookup shape (`[vocab, dim]` + arbitrary-shape indices → stacked rows), not general indexing.

## Core Abstractions

- `NDArray`: The primary data structure for N-dimensional arrays.
- `Shape`: Immutable representation of array dimensions, with NumPy-style broadcasting (`Shape.broadcast`).
- `JAX`: Entry point for transformations (`jit`, `grad`, `gradTree`, `vmap`).
- `Jaxpr`: A sequence of primitive equations representing a traced function.
- `PyTree`: A nested `List`/`Map<String, _>` container of `NDArray` leaves (see `com.marmanis.jax4j.pytree`).

## Usage Example: Linear Regression

```java
// Define a loss function
Function<NDArray, NDArray> lossFn = w -> {
    NDArray prediction = x.mul(w).add(bias);
    NDArray diff = prediction.sub(y);
    return diff.mul(diff).mean();
};

// Compute gradients automatically
Function<NDArray, NDArray> gradFn = JAX.grad(lossFn);

// Optimization loop
for (int i = 0; i < 100; i++) {
    NDArray grad = gradFn.apply(w);
    w = w.sub(grad.mul(learningRate));
}
```

## Usage Example: Gradients over a Pytree of Parameters

```java
Map<String, PyTree> fields = new LinkedHashMap<>();
fields.put("w", PyTree.leaf(w));
fields.put("b", PyTree.leaf(b));
PyTree params = PyTree.dict(fields);

Function<PyTree, NDArray> loss = p -> {
    List<NDArray> leaves = PyTrees.flatten(p);
    NDArray pred = x.mul(leaves.get(0)).add(leaves.get(1));
    return pred.sub(y).mul(pred.sub(y)).mean();
};

PyTree grads = JAX.gradTree(loss).apply(params);
params = PyTrees.map2(NDArray::sub, params, PyTrees.map(g -> g.mul(scalar(learningRate)), grads));
```

## Usage Example: Batching with `vmap`

```java
// Function written for a single example...
Function<NDArray, NDArray> predict = x -> x.dot(weights).sum();

// ...automatically vectorized over a leading batch dimension.
NDArray batchOfPredictions = JAX.vmap(predict).apply(batchOfExamples);
```

## Usage Example: Explicit Device Placement and Multi-Device Dispatch

```java
// Place two arrays on the same explicit device; matching-shape elementwise
// ops on them dispatch to that device via TornadoVM (falling back to the
// host automatically if no TornadoVM runtime is configured).
Device gpu = Device.defaultDevice();
NDArray a = x.to(gpu);
NDArray b = y.to(gpu);
NDArray result = a.mul(b); // runs on gpu

// Fan out independent work across several devices concurrently.
NDArray[] shardResults = MultiDevice.map(devices, shards, shard -> shard.mul(shard));
```

## Setup and Requirements

- **JDK**: Java 24 or higher.
- **TornadoVM**: Ensure the TornadoVM SDK is installed and configured in your environment to enable hardware acceleration.
- **Maven**: Standard Maven build system.

### Selecting a TornadoVM backend in the examples

The runnable examples in `com.marmanis.jax4j.examples` (`LinearRegression`, `MNISTFromScratch`)
default to running on TornadoVM's PTX backend (the discrete NVIDIA GPU, when one is
discovered). Pass `--backend=opencl` to run on the multi-core CPU via TornadoVM's OpenCL/POCL
backend instead:

```bash
mvn exec:java -Dexec.mainClass=com.marmanis.jax4j.examples.LinearRegression
mvn exec:java -Dexec.mainClass=com.marmanis.jax4j.examples.LinearRegression -Dexec.args=--backend=opencl
```

This calls `Device.useBackend("ptx" | "opencl")`, which falls back to the host CPU with a
warning if the requested backend has no discoverable device (e.g. no TornadoVM runtime
configured, or a single-backend SDK).

## Measured GPU acceleration: use with eyes open

FLOAT64 kernels exist for every elementwise op and matmul
(`TornadoVMBackend`), and are correct on the OpenCL backend. Whether you
actually get a speedup depends heavily on the hardware and the workload.
For the sibling [chebfun4j](../chebfun4j) project, which drove FP64 to be
added at all, the FP64 GPU path was measured to be **slower than the host
CPU** in every case tested — the details matter enough to record here.

**Hardware measured:** NVIDIA GeForce RTX 3050 Laptop (Ampere), 4 GB VRAM,
PCIe Gen 3. **Test:** 1 M-element FP64 vectors and a 128×128 FP64 matmul,
via `com.marmanis.jax4j.examples.Fp64GpuVerify`.

| Op | Host CPU | OpenCL | PTX | Correct? |
|---|---|---|---|---|
| ADD | 11 ms | 629 ms | 404 ms | ✓ both |
| MUL | 16 ms | 156 ms | 47 ms | ✓ both |
| SUB | 8 ms | 149 ms | 43 ms | ✓ both |
| DIV | 8 ms | 133 ms | 47 ms | ✓ both |
| EXP | 8 ms | 121 ms | 39 ms | ✓ OpenCL; **✗ PTX (5.2e-5 err)** |
| SIN | 8 ms | 144 ms | 29 ms | ✓ OpenCL; **✗ PTX (7.4e-7 err)** |
| COS | 11 ms | 134 ms | 30 ms | ✓ OpenCL; **✗ PTX (5.8e-7 err)** |
| TANH | 13 ms | 139 ms | 29 ms | ✓ OpenCL; **✗ PTX (7.9e-6 err)** |
| matmul 128² | 6 ms | 168 ms | 48 ms | ✓ both |

Three things drive these numbers:

1. **Consumer NVIDIA cards throttle FP64 to 1/32 – 1/64 of FP32 throughput.**
   Datacenter cards (A100, H100) don't; the same code should look very
   different on those. This is a hardware / SKU issue, not a jax4j one.
2. **`DataTransferMode.EVERY_EXECUTION`** ships the input and output arrays
   to and from GPU on every primitive call. 8 MB per FP64 vector × PCIe Gen 3
   ≈ 20 ms of pure overhead per op — already more than the CPU total on
   several ops. A future `NDArray` lifecycle redesign around persistent
   device residency (`FIRST_EXECUTION` transfer mode) would remove this
   floor, but the arithmetic ceiling stays.
3. **PTX transcendentals appear to be emitted as FP32-approximate PTX
   instructions on FP64 inputs.** The observed accuracy of EXP/SIN/COS/TANH
   on PTX matches `ex2.approx.f32` / `sin.approx.f32` etc. exactly. Reported
   upstream at `https://github.com/beehive-lab/TornadoVM/issues/...` (fill
   in once triaged). Until fixed, do not use the PTX backend for anything
   that needs more than ~6-digit accuracy on `Math.exp/sin/cos/tanh`.

**Practical guidance.** For FLOAT64 workloads on a consumer NVIDIA GPU
where kernels do a single elementwise pass over the data, the host CPU
wins — set `.to(Device.host())` and skip the device path. For genuine
speedup with the FP64 kernels shipped here you'll want datacenter FP64
throughput and long kernel chains that keep data on the device across many
ops. The FP32 kernels remain a much more favourable case and are wired up
in every `ConcreteNDArray` path already.

## Project Structure

- `com.marmanis.jax4j.core`: Fundamental data structures.
- `com.marmanis.jax4j.ir`: Intermediate Representation (Jaxpr).
- `com.marmanis.jax4j.tracing`: Tracing system for capturing Java functions.
- `com.marmanis.jax4j.api`: High-level JAX transformations (`JAX`, `Grad`, `Vmap`, `FFI`).
- `com.marmanis.jax4j.pytree`: Pytree containers and flatten/unflatten/map utilities.
- `com.marmanis.jax4j.backend`: TornadoVM compilation and execution.

## Testing

Run the test suite using Maven:
```bash
mvn test
```

The tests cover:
- Tracing correctness.
- Automatic differentiation accuracy (via `GradTest`).
- NDArray arithmetic.

## License

This project is licensed under the Apache License 2.0.
