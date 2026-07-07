package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.FFI;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Arrays;

/**
 * Examples for the jax4j Foreign Function Interface.
 */
public class FFIExamples {

    public static void main(String[] args) {
        setupFFITargets();

        // 1. RMS Norm Example
        System.out.println("--- RMS Norm Example ---");
        NDArray x = new ConcreteNDArray(new float[]{1, 2, 3, 4}, new Shape(4));
        float eps = 1e-5f;
        // Attributes can be passed as additional arguments or via a configuration object
        NDArray rmsResult = FFI.call("rms_norm", x.shape(), x, scalar(eps));
        System.out.println("Input: " + x);
        System.out.println("RMS Result: " + rmsResult);

        // 2. CPU Example (Simple vector multiplication)
        System.out.println("\n--- CPU Example ---");
        NDArray cpuResult = FFI.call("cpu_mul", x.shape(), x, x);
        System.out.println("CPU Mul Result: " + cpuResult);

        // 3. CUDA Example (Accelerated via TornadoVM)
        System.out.println("\n--- CUDA Example (via TornadoVM) ---");
        NDArray cudaResult = FFI.call("cuda_add", x.shape(), x, x);
        System.out.println("CUDA Add Result: " + cudaResult);
    }

    public static void setupFFITargets() {
        // Implementation of RMS Norm (Pure Java for demonstration, could be native via Panama)
        FFI.register("rms_norm", inputs -> {
            NDArray x = inputs[0];
            float eps = inputs[1].toFloatArray()[0];
            float[] data = x.toFloatArray();
            float sumSq = 0;
            for (float v : data) sumSq += v * v;
            float invRms = (float) (1.0 / Math.sqrt(sumSq / data.length + eps));
            float[] out = new float[data.length];
            for (int i = 0; i < data.length; i++) out[i] = data[i] * invRms;
            return new ConcreteNDArray(out, x.shape());
        });

        // Implementation of a custom CPU multiplier
        FFI.register("cpu_mul", inputs -> {
            NDArray a = inputs[0];
            NDArray b = inputs[1];
            return a.mul(b);
        });

        // Implementation of a custom CUDA-accelerated adder using TornadoVM
        FFI.register("cuda_add", inputs -> {
            try {
                return runCudaAdd(inputs);
            } catch (Throwable t) {
                System.err.println("WARNING: TornadoVM execution failed, falling back to CPU: " + t.getMessage());
                float[] a = inputs[0].toFloatArray();
                float[] b = inputs[1].toFloatArray();
                float[] c = new float[a.length];
                for (int i = 0; i < a.length; i++) c[i] = a[i] + b[i];
                return new ConcreteNDArray(c, inputs[0].shape());
            }
        });
    }

    private static NDArray runCudaAdd(NDArray[] inputs) {
        float[] a = inputs[0].toFloatArray();
        float[] b = inputs[1].toFloatArray();
        float[] c = new float[a.length];

        TaskGraph tg = new TaskGraph("ffi_cuda")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b)
            .task("add", FFIExamples::vectorAdd, a, b, c)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
        
        new TornadoExecutionPlan(tg.snapshot()).execute();
        return new ConcreteNDArray(c, inputs[0].shape());
    }

    public static void vectorAdd(float[] a, float[] b, float[] c) {
        for (@Parallel int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    private static NDArray scalar(float v) {
        return new ConcreteNDArray(new float[]{v}, new Shape(1));
    }
}
