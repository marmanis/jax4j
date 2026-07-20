package com.marmanis.jax4j.backend;

import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.AxisMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.*;

public class TornadoJitCompiler {
    private static final Logger log = LoggerFactory.getLogger(TornadoJitCompiler.class);

    public static class CompiledPlan {
        private final TornadoExecutionPlan plan;
        private final Map<Integer, float[]> arrays;
        private final List<Var> inVars;
        private final List<Var> outVars;
        private final List<MeanPostProcess> meanPostProcesses;

        public CompiledPlan(TornadoExecutionPlan plan, Map<Integer, float[]> arrays, List<Var> inVars, List<Var> outVars, List<MeanPostProcess> meanPostProcesses) {
            this.plan = plan;
            this.arrays = arrays;
            this.inVars = inVars;
            this.outVars = outVars;
            this.meanPostProcesses = meanPostProcesses;
        }

        public NDArray execute(List<NDArray> args, Device device) {
            // 1. Copy concrete inputs to JVM JIT-allocated arrays.
            for (int i = 0; i < args.size(); i++) {
                NDArray arg = args.get(i);
                float[] dest = arrays.get(inVars.get(i).id());
                float[] srcData = arg.toFloatArray();
                System.arraycopy(srcData, 0, dest, 0, srcData.length);
            }

            // 2. Execute on GPU.
            try {
                plan.execute();
            } catch (Exception e) {
                throw new RuntimeException("TornadoVM plan execution failed", e);
            }

            // 3. Post-process any MEAN reductions.
            for (MeanPostProcess mpp : meanPostProcesses) {
                float[] out = arrays.get(mpp.outId);
                out[0] /= mpp.count;
            }

            // 4. Retrieve final output array.
            Var outVar = outVars.get(0);
            float[] outData = arrays.get(outVar.id());
            // Create a copy of the output array.
            float[] res = new float[outData.length];
            System.arraycopy(outData, 0, res, 0, outData.length);
            return new ConcreteNDArray(res, outVar.shape(), outVar.dtype(), device);
        }

        public void close() {
            try {
                plan.close();
            } catch (Exception e) {
                log.warn("Failed to close JIT plan: {}", e.getMessage());
            }
        }
    }

    private record MeanPostProcess(int outId, int count) {}

    public static CompiledPlan compile(Jaxpr jaxpr, Device device) {
        // If not a TornadoVM device (e.g. host), don't compile.
        if (device.getTornadoDevice() == null) {
            throw new IllegalArgumentException("Cannot JIT compile to host device via TornadoVM");
        }

        Map<Integer, float[]> arrays = new HashMap<>();

        // Allocate float[] arrays for every variable in the jaxpr.
        // Input variables
        for (Var v : jaxpr.inVars()) {
            if (v.dtype() != DType.FLOAT32 && v.dtype() != DType.BOOL && v.dtype() != DType.INT32) {
                throw new UnsupportedOperationException("TornadoVM JIT compiler only supports FLOAT32, BOOL, and INT32 variables");
            }
            arrays.put(v.id(), new float[(int) v.shape().size()]);
        }

        // Constants
        for (Map.Entry<Integer, NDArray> entry : jaxpr.consts().entrySet()) {
            NDArray val = entry.getValue();
            float[] data = val.toFloatArray();
            arrays.put(entry.getKey(), data);
        }

        // Equations outputs
        for (Equation eq : jaxpr.equations()) {
            for (Var v : eq.outputs()) {
                arrays.put(v.id(), new float[(int) v.shape().size()]);
            }
        }

        TaskGraph tg = new TaskGraph("jit_" + jaxpr.hashCode());

        // 1. Declare input transfers
        for (Var v : jaxpr.inVars()) {
            tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, arrays.get(v.id()));
        }

        // 2. Declare constant transfers (only on first execution!)
        for (Integer id : jaxpr.consts().keySet()) {
            tg.transferToDevice(DataTransferMode.FIRST_EXECUTION, arrays.get(id));
        }

        // 2b. Identify outVars IDs for quick lookup
        Set<Integer> outVarIds = new HashSet<>();
        for (Var v : jaxpr.outVars()) {
            outVarIds.add(v.id());
        }

        // Declare intermediate transfers (so TornadoVM registers their arrays)
        for (Equation eq : jaxpr.equations()) {
            for (Var v : eq.outputs()) {
                if (!outVarIds.contains(v.id())) {
                    tg.transferToDevice(DataTransferMode.FIRST_EXECUTION, arrays.get(v.id()));
                }
            }
        }

        List<MeanPostProcess> meanPostProcesses = new ArrayList<>();

        // 3. Add tasks for each equation
        int taskCount = 0;
        for (Equation eq : jaxpr.equations()) {
            String taskName = "t" + (taskCount++);
            Primitive p = eq.primitive();
            Var outVar = eq.outputs().get(0);
            float[] outArr = arrays.get(outVar.id());

            switch (p) {
                case ADD -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorAdd, a, b, outArr);
                }
                case SUB -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorSub, a, b, outArr);
                }
                case MUL -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorMul, a, b, outArr);
                }
                case DIV -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorDiv, a, b, outArr);
                }
                case MAX -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorMax, a, b, outArr);
                }
                case MIN -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    tg.task(taskName, TornadoVMBackend::vectorMin, a, b, outArr);
                }
                case EXP -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorExp, a, outArr);
                }
                case LOG -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorLog, a, outArr);
                }
                case SIN -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorSin, a, outArr);
                }
                case COS -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorCos, a, outArr);
                }
                case TANH -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorTanh, a, outArr);
                }
                case RELU -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorRelu, a, outArr);
                }
                case SIGMOID -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorSigmoid, a, outArr);
                }
                case DOT -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    float[] b = arrays.get(eq.inputs().get(1).id());
                    int M = eq.inputs().get(0).shape().dimensions()[0];
                    int K = eq.inputs().get(0).shape().dimensions()[1];
                    int N = eq.inputs().get(1).shape().dimensions()[1];
                    tg.task(taskName, TornadoVMBackend::matmulKernel, a, b, outArr, M, K, N);
                }
                case SUM -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::reduceSum, a, outArr);
                }
                case MEAN -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::reduceSum, a, outArr);
                    meanPostProcesses.add(new MeanPostProcess(outVar.id(), a.length));
                }
                case CAST -> {
                    float[] a = arrays.get(eq.inputs().get(0).id());
                    tg.task(taskName, TornadoVMBackend::vectorCopy, a, outArr);
                }
                default -> throw new UnsupportedOperationException("TornadoVM JIT compiler does not support primitive: " + p);
            }
        }

        // 4. Declare output transfers
        for (Var v : jaxpr.outVars()) {
            tg.transferToHost(DataTransferMode.EVERY_EXECUTION, arrays.get(v.id()));
        }

        ImmutableTaskGraph itg = tg.snapshot();
        TornadoExecutionPlan plan = new TornadoExecutionPlan(itg);
        if (device.getTornadoDevice() != null) {
            plan.withDevice(device.getTornadoDevice());
        }

        return new CompiledPlan(plan, arrays, jaxpr.inVars(), jaxpr.outVars(), meanPostProcesses);
    }
}
