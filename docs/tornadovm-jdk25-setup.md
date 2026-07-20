# TornadoVM on JDK 25 — setup and benchmarking

This is a run-through for wiring up TornadoVM GPU acceleration on JDK 25 and
running [`BackendBenchmark`](../src/main/java/com/marmanis/jax4j/examples/BackendBenchmark.java)
to compare single-core CPU, multi-core CPU, CPU-via-OpenCL, and NVIDIA GPU.

## What you need on disk

- **JDK 25** (Corretto, Temurin, or Oracle — anything current). `JAVA_HOME` must point at it.
- **A JDK 25 build of TornadoVM 4.0.1.** The current Maven dependencies (`tornado-api`, `tornado-matrices`) already resolve to `4.0.1-jdk25`, so you need the matching runtime SDK. Two ways to get it:

  1. **Prebuilt binary** from the TornadoVM releases page:
     - https://github.com/beehive-lab/TornadoVM/releases
     - Look for a file named like `tornadovm-4.0.1-jdk25-*-opencl-ptx-windows-amd64.zip` (or `-linux-amd64.tar.gz` on Linux). Extract somewhere stable, e.g. `C:\Code\GitHub\Java\TornadoVM\dist\`.

  2. **Build from source** with the TornadoVM installer:
     ```bash
     git clone https://github.com/beehive-lab/TornadoVM
     cd TornadoVM
     ./bin/tornadovm-installer --jdk jdk25 --backend opencl,ptx
     ```
     Requires CUDA (for PTX) and an OpenCL runtime (for OpenCL). The installer places the resulting SDK under `TornadoVM/dist/`.

## Environment variables

Point `TORNADOVM_HOME` at the extracted/built SDK **directory that contains** `bin/`, `lib/`, `share/java/`, `etc/exportLists/`:

```powershell
# PowerShell (permanent)
[Environment]::SetEnvironmentVariable(
    'TORNADOVM_HOME',
    'C:\Code\GitHub\Java\TornadoVM\dist\tornadovm-4.0.1-jdk25-opencl-ptx-windows-amd64\tornadovm-4.0.1-jdk25-opencl-ptx',
    'User')
```

The Maven `tornado` profile in both [jax4j/pom.xml](../pom.xml) and [chebfun4j/pom.xml](../../chebfun4j/pom.xml) auto-activates when this variable is set. Verify:

```bash
mvn help:active-profiles       # should list "tornado"
```

If you want to *temporarily* disable the profile (e.g. run tests without TornadoVM args on a machine where the SDK is broken), use `-P'!tornado'`:

```bash
mvn -P'!tornado' test
```

## GPU / OpenCL device availability

**NVIDIA GPU** requires the NVIDIA CUDA toolkit installed (for TornadoVM's PTX backend) *or* the NVIDIA OpenCL runtime (comes with the display driver on Windows). TornadoVM exposes NVIDIA cards through both backends.

**CPU as an OpenCL device** requires a CPU OpenCL runtime — none is installed by default on Windows. Options:

- **POCL** (Portable Computing Language) — open-source, exposes any x86 CPU as an OpenCL device. Recommended. Prebuilt Windows binaries are available; on Linux install via package manager.
- **Intel OpenCL runtime** — if you have an Intel CPU; ships with Intel oneAPI.
- **NVIDIA's OpenCL runtime does not expose CPU devices**, only the GPU.

Verify what TornadoVM sees after setup:

```bash
# Prints all discovered devices, one line per driver/slot.
$TORNADOVM_HOME/bin/tornado --devices
```

Expected output on the target machine (RTX 3050 + POCL installed):

```
Number of Tornado drivers: 2
Driver: OpenCL
  Total number of OpenCL devices  : 2
    Tornado device=0:0  (NVIDIA CUDA)  NVIDIA GeForce RTX 3050 Laptop GPU
    Tornado device=0:1  (Portable Computing Language)  <your CPU model>
Driver: PTX
  Total number of PTX devices  : 1
    Tornado device=1:0  NVIDIA GeForce RTX 3050 Laptop GPU
```

## Running the benchmark

From the workspace root or `jax4j/`:

```bash
mvn -Ptornado exec:exec -Dexec.mainClass=com.marmanis.jax4j.examples.BackendBenchmark
```

The tornado profile switches the exec plugin to `exec:exec` (which forks a new JVM with the full TornadoVM argLine — required because TornadoVM needs `--module-path` set at JVM start).

### What it measures

Wall-clock median (of 5 timed runs, after 2 warm-ups) per (op, size) row, across every discovered target. Columns:

| column | source |
| --- | --- |
| `CPU-1` | Single-thread `HostBackend` loop |
| `CPU-N` | Java parallel streams (the same path `ConcreteNDArray` takes above `PARALLEL_THRESHOLD = 65536`) |
| `NVIDIA GeForce ... (GPU)` | TornadoVM PTX or OpenCL to the GPU |
| `<your CPU> (CPU)` | TornadoVM OpenCL to the CPU via POCL/Intel runtime — only appears if a CPU OpenCL device is discovered |

Rows: elementwise `ADD`, `MUL`, `EXP` at sizes 4 096 / 65 536 / 1 048 576, and `MATMUL` at 64² / 128² / 256².

### Interpreting results

Broadly, from the existing FP64 measurements in the [main README](../README.md#measured-gpu-acceleration-use-with-eyes-open):

- **Small ops** (below the parallel threshold): `CPU-1` and `CPU-N` are indistinguishable; TornadoVM devices lose to both because PCIe transfer time dominates.
- **Medium/large elementwise**: `CPU-N` pulls ahead of `CPU-1` roughly by core count; GPU should beat both if the FP32 arithmetic intensity is high enough (transcendentals like `EXP`) but may still lose on pure add/mul if transfer time dominates.
- **Matmul**: this is where the GPU should win at 256²+, because the O(n³) work per O(n²) data amortizes the transfer.

The `maxErr` column is the worst absolute deviation between each target's output and the sequential host baseline. FP32 add/mul/sub should be bit-identical (`0.00e+00`); FP32 div can have small last-bit differences; transcendentals (EXP/SIN/COS/TANH) differ per platform's math library and can drift up to ~1e-4 on some PTX runtimes.

## Troubleshooting

- **`Could not initialize class TornadoExecutionPlan`** — TornadoVM args not on the JVM at start. Check that `mvn help:active-profiles` lists `tornado`, and that you're using `exec:exec` (not `exec:java`) so a fresh JVM gets forked with the argLine.
- **`Error occurred during initialization of boot layer`** — a JDK 21 TornadoVM SDK is being used with JDK 25 (typical symptom: `--enable-preview` in the argLine, or module descriptors that JDK 25 rejects). Update `TORNADOVM_HOME` to point at a JDK 25 SDK.
- **Only `CPU-1` and `CPU-N` columns in the benchmark output** — TornadoVM discovered zero devices. Either the SDK isn't found (check `$TORNADOVM_HOME/bin/tornado --devices`) or no GPU/CPU-OpenCL runtime is installed.
- **CPU column missing** — POCL (or another CPU OpenCL runtime) isn't installed. The GPU column should still appear.
