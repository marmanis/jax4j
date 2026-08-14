package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Modules;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.scalar;
import static com.marmanis.jax4j.testutil.GradChecker.vector;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Equinox-style dynamic parameter partitioning and freezing.
 */
public class ModulesPartitionTest {

    public record SimpleModel(NDArray weight, NDArray bias) implements Module {}

    @Test
    public void testDynamicPartitionAndCombine() {
        NDArray w = vector(1.0f, 2.0f);
        NDArray b = scalar(5.0f);
        SimpleModel model = new SimpleModel(w, b);

        // Partition: only allow weights to be trainable (filter weights)
        PyTree paramTree = Modules.flatten(model, arr -> arr == w);
        List<NDArray> leaves = PyTrees.flatten(paramTree);

        assertEquals(2, leaves.size());
        assertEquals(w, leaves.get(0));
        assertNull(leaves.get(1)); // bias is frozen (null)

        // Combine/unflatten using a modified weight but original template
        NDArray newW = vector(10.0f, 20.0f);
        PyTree updatedTree = PyTree.dict(java.util.Map.of(
            "weight", PyTree.leaf(newW),
            "bias", PyTree.leaf(null)
        ));

        SimpleModel updatedModelParams = Modules.unflatten(model, updatedTree);
        assertNull(updatedModelParams.bias()); // bias is null in the params representation

        SimpleModel updatedModel = Modules.combine(updatedModelParams, model);
        assertEquals(newW, updatedModel.weight());
        assertEquals(b, updatedModel.bias()); // bias preserved from the original template
    }

    @Test
    public void testDynamicFilterGradEager() {
        NDArray w = vector(1.0f, 2.0f);
        NDArray b = scalar(5.0f);
        SimpleModel model = new SimpleModel(w, b);

        // Loss function: sum((x * w + b)^2)
        NDArray x = vector(2.0f, 3.0f);
        Function<SimpleModel, NDArray> lossFn = m -> {
            NDArray pred = x.mul(m.weight()).add(m.bias());
            return pred.mul(pred).sum();
        };

        // Train weights only, freeze bias
        SimpleModel grads = Modules.filterGrad(lossFn, model, arr -> arr == w);

        assertNotNull(grads.weight());
        assertNull(grads.bias()); // gradient is null (frozen)

        // Expected gradient for weight: dL/dw = 2 * (x * w + b) * x
        float[] wData = w.toFloatArray();
        float[] xData = x.toFloatArray();
        float bData = b.toFloatArray()[0];
        float[] expectedWeightGrad = new float[2];
        for (int i = 0; i < 2; i++) {
            float pred = xData[i] * wData[i] + bData;
            expectedWeightGrad[i] = 2 * pred * xData[i];
        }

        assertArrayEquals(expectedWeightGrad, grads.weight().toFloatArray(), 1e-4f);
    }

    @Test
    public void testDynamicFilterGradJit() {
        NDArray w = vector(1.0f, 2.0f);
        NDArray b = scalar(5.0f);
        SimpleModel model = new SimpleModel(w, b);

        NDArray x = vector(2.0f, 3.0f);
        Function<SimpleModel, NDArray> lossFn = m -> {
            NDArray pred = x.mul(m.weight()).add(m.bias());
            return pred.mul(pred).sum();
        };

        // Trace and JIT compile gradient with dynamic filter: only weights are trainable
        Function<PyTree, PyTree> jitGradFn = JAX.jitGradTree(params -> {
            SimpleModel m = Modules.combine(Modules.unflatten(model, params), model);
            return lossFn.apply(m);
        });

        // Dynamic partition tree
        PyTree paramTree = Modules.flatten(model, arr -> arr == w);

        PyTree gradsTree = jitGradFn.apply(paramTree);
        SimpleModel grads = Modules.unflatten(model, gradsTree);

        assertNotNull(grads.weight());
        assertNull(grads.bias()); // gradient is null (frozen)

        // Repeat with updated weights to test JIT cache hit
        NDArray w2 = vector(0.5f, 1.5f);
        PyTree paramTree2 = Modules.flatten(new SimpleModel(w2, b), arr -> arr == w2);
        PyTree gradsTree2 = jitGradFn.apply(paramTree2);
        SimpleModel grads2 = Modules.unflatten(model, gradsTree2);

        assertNotNull(grads2.weight());
        assertNull(grads2.bias());

        float[] w2Data = w2.toFloatArray();
        float[] xData = x.toFloatArray();
        float bData = b.toFloatArray()[0];
        float[] expectedWeightGrad2 = new float[2];
        for (int i = 0; i < 2; i++) {
            float pred = xData[i] * w2Data[i] + bData;
            expectedWeightGrad2[i] = 2 * pred * xData[i];
        }

        assertArrayEquals(expectedWeightGrad2, grads2.weight().toFloatArray(), 1e-4f);
    }
}
