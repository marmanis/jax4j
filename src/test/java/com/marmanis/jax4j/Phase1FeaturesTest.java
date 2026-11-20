package com.marmanis.jax4j;

import com.marmanis.jax4j.api.AuxResult;
import com.marmanis.jax4j.api.CustomVjp;
import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.PRNGSequence;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Modules;
import com.marmanis.jax4j.ml.nn.Linear;
import com.marmanis.jax4j.ml.nn.RMSNorm;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for Phase 1 enhancements:
 * - has_aux autodiff support (both NDArray and PyTree)
 * - CustomVjp fluent builder
 * - PRNGSequence key generator
 * - RMSNorm normalization module
 */
public class Phase1FeaturesTest {

    @Test
    public void testValueAndGradAuxNDArray() {
        // fn(x) returns x^2 + 3*x and auxiliary string "metric_42"
        Function<NDArray, AuxResult<NDArray, String>> fn = (x) -> {
            NDArray val = x.mul(x).add(x.mul(new ConcreteNDArray(new float[]{3f}, new Shape(1))));
            return AuxResult.of(val.sum(), "loss_metadata_ok");
        };

        NDArray x = new ConcreteNDArray(new float[]{2.0f}, new Shape(1));
        AuxResult<NDArray[], String> result = JAX.value_and_grad(fn, true).apply(x);

        assertEquals("loss_metadata_ok", result.aux());
        NDArray[] valAndGrad = result.value();
        // val = 2^2 + 3*2 = 10
        assertEquals(10.0f, valAndGrad[0].toFloatArray()[0], 1e-4f);
        // grad = 2*x + 3 = 2*2 + 3 = 7
        assertEquals(7.0f, valAndGrad[1].toFloatArray()[0], 1e-4f);

        // Also test grad(fn, true)
        AuxResult<NDArray, String> gradOnly = JAX.grad(fn, true).apply(x);
        assertEquals("loss_metadata_ok", gradOnly.aux());
        assertEquals(7.0f, gradOnly.value().toFloatArray()[0], 1e-4f);
    }

    @Test
    public void testGradTreeAux() {
        PRNGKey key = PRNGKey.key(42);
        Linear linear = Linear.init(key, 2, 1);

        // Model returns loss + auxiliary record
        record TrainingMetrics(float sampleCount, String status) {}

        Function<PyTree, AuxResult<NDArray, TrainingMetrics>> lossFn = (params) -> {
            Linear m = Modules.unflatten(linear, params);
            NDArray input = new ConcreteNDArray(new float[]{1.0f, 2.0f}, new Shape(1, 2));
            NDArray target = new ConcreteNDArray(new float[]{5.0f}, new Shape(1, 1));
            NDArray pred = m.apply(input);
            NDArray loss = pred.sub(target).mul(pred.sub(target)).sum();
            return AuxResult.of(loss, new TrainingMetrics(1.0f, "BATCH_PROCESSED"));
        };

        PyTree params = Modules.flatten(linear);
        AuxResult<PyTree, TrainingMetrics> gradResult = Grad.gradTree_aux(lossFn).apply(params);

        assertNotNull(gradResult.aux());
        assertEquals(1.0f, gradResult.aux().sampleCount());
        assertEquals("BATCH_PROCESSED", gradResult.aux().status());

        List<NDArray> gradLeaves = PyTrees.flatten(gradResult.value());
        assertEquals(2, gradLeaves.size()); // w and b
        assertNotNull(gradLeaves.get(0));
        assertNotNull(gradLeaves.get(1));
    }

    @Test
    public void testCustomVjpBuilder() {
        // Forward: y = x * 2
        // Backward override: instead of grad = 2 * g, return grad = 100 * g
        Function<NDArray, NDArray> fn = CustomVjp.builder((NDArray x) -> x.mul(new ConcreteNDArray(new float[]{2f}, new Shape(1))))
                .defvjp((NDArray x, NDArray g) -> g.mul(new ConcreteNDArray(new float[]{100f}, new Shape(1))))
                .build();

        NDArray x = new ConcreteNDArray(new float[]{3.0f}, new Shape(1));
        NDArray grad = Grad.grad((arg) -> fn.apply(arg).sum()).apply(x);

        assertEquals(100.0f, grad.toFloatArray()[0], 1e-4f);
    }

    @Test
    public void testPRNGSequence() {
        PRNGSequence seq1 = new PRNGSequence(100L);
        PRNGSequence seq2 = new PRNGSequence(100L);

        PRNGKey k1_1 = seq1.next();
        PRNGKey k1_2 = seq1.next();

        PRNGKey k2_1 = seq2.next();
        PRNGKey k2_2 = seq2.next();

        // Reproducibility across identical seeds
        assertEquals(k1_1.state(), k2_1.state());
        assertEquals(k1_2.state(), k2_2.state());

        // Independence between consecutive keys
        assertNotEquals(k1_1.state(), k1_2.state());

        // Batch generation
        PRNGKey[] batch = seq1.next(5);
        assertEquals(5, batch.length);
    }

    @Test
    public void testRMSNormForwardAndGrad() {
        int dim = 4;
        RMSNorm norm = RMSNorm.init(dim, 1e-5f);

        // Input vector [1.0, 2.0, 3.0, 4.0]
        // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
        // rms = sqrt(7.5) ≈ 2.738612787
        NDArray x = new ConcreteNDArray(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new Shape(1, dim));
        NDArray y = norm.apply(x);

        float[] yData = y.toFloatArray();
        float expectedRms = (float) Math.sqrt(7.5);
        for (int i = 0; i < dim; i++) {
            assertEquals((i + 1) / expectedRms, yData[i], 1e-3f);
        }

        // Check autograd through RMSNorm
        PyTree params = Modules.flatten(norm);
        PyTree grads = Grad.gradTree((p) -> {
            RMSNorm m = Modules.unflatten(norm, p);
            return m.apply(x).sum();
        }).apply(params);

        List<NDArray> gradLeaves = PyTrees.flatten(grads);
        assertEquals(1, gradLeaves.size());
        assertNotNull(gradLeaves.get(0));
        assertEquals(new Shape(dim), gradLeaves.get(0).shape());
    }
}
