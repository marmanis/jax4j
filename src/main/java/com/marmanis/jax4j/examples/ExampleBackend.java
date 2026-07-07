package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.core.Device;

/**
 * Shared command-line handling for the {@code --backend=opencl|ptx} option
 * every example accepts. Defaults to {@code ptx} (the discrete NVIDIA GPU in
 * this environment); pass {@code --backend=opencl} to instead run on the
 * multi-core CPU via TornadoVM's OpenCL/POCL backend.
 */
final class ExampleBackend {
    private ExampleBackend() {}

    static void selectFromArgs(String[] args) {
        String backend = "ptx";
        for (String arg : args) {
            if (arg.startsWith("--backend=")) {
                backend = arg.substring("--backend=".length());
            }
        }
        Device.useBackend(backend);
        System.out.println("Running on TornadoVM backend '" + backend + "': " + Device.defaultDevice());
    }
}
