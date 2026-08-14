package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Modules;
import com.marmanis.jax4j.ml.nn.Conv2d;
import com.marmanis.jax4j.ml.nn.LSTM;
import com.marmanis.jax4j.ml.nn.GRU;
import com.marmanis.jax4j.ml.nn.MultiHeadAttention;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Tests for new neural network layers: Conv2d, LSTM, GRU, and MultiHeadAttention.
 */
public class NnLayersTest {

    @Test
    public void testConv2dForwardAndGrad() {
        PRNGKey key = PRNGKey.key(12345);
        // Conv2d: inC=1, outC=1, kH=3, kW=3, stride=1
        Conv2d conv = Conv2d.init(key, 1, 1, 3, 3, 1);

        // Input shape: [1, 5, 5]
        float[] inputData = new float[25];
        java.util.Arrays.fill(inputData, 1.0f);
        NDArray input = new ConcreteNDArray(inputData, new Shape(1, 5, 5));

        NDArray out = conv.apply(input);
        assertEquals(new Shape(1, 3, 3), out.shape());

        // Test batched input [2, 1, 5, 5]
        NDArray batchedInput = new ConcreteNDArray(new float[50], new Shape(2, 1, 5, 5));
        NDArray batchedOut = conv.apply(batchedInput);
        assertEquals(new Shape(2, 1, 3, 3), batchedOut.shape());

        // Test grad
        Function<PyTree, NDArray> lossFn = t -> {
            Conv2d m = Modules.unflatten(conv, t);
            return m.apply(input).sum();
        };

        PyTree params = Modules.flatten(conv);
        PyTree gradsTree = Grad.gradTree(lossFn).apply(params);
        Conv2d grads = Modules.unflatten(conv, gradsTree);

        assertNotNull(grads.weight());
        assertNotNull(grads.bias());
        assertEquals(new Shape(1, 1, 3, 3), grads.weight().shape());
        assertEquals(new Shape(1), grads.bias().shape());
    }

    @Test
    public void testLSTMSequenceForwardAndGrad() {
        PRNGKey key = PRNGKey.key(12345);
        // LSTM: inputSize=3, hiddenSize=4
        LSTM lstm = LSTM.init(key, 3, 4);

        // Inputs: [5 steps, 3 features]
        NDArray xs = new ConcreteNDArray(new float[15], new Shape(5, 3));
        NDArray initH = new ConcreteNDArray(new float[4], new Shape(4));
        NDArray initC = new ConcreteNDArray(new float[4], new Shape(4));

        NDArray ys = lstm.applySequence(xs, initH, initC);
        assertEquals(new Shape(5, 4), ys.shape());

        // Test grad
        Function<PyTree, NDArray> lossFn = t -> {
            LSTM m = Modules.unflatten(lstm, t);
            return m.applySequence(xs, initH, initC).sum();
        };

        PyTree params = Modules.flatten(lstm);
        PyTree gradsTree = Grad.gradTree(lossFn).apply(params);
        LSTM grads = Modules.unflatten(lstm, gradsTree);

        assertNotNull(grads.wGates());
        assertNotNull(grads.uGates());
        assertNotNull(grads.bGates());
        assertEquals(new Shape(16, 3), grads.wGates().shape());
        assertEquals(new Shape(16, 4), grads.uGates().shape());
        assertEquals(new Shape(16), grads.bGates().shape());
    }

    @Test
    public void testGRUSequenceForwardAndGrad() {
        PRNGKey key = PRNGKey.key(12345);
        // GRU: inputSize=3, hiddenSize=4
        GRU gru = GRU.init(key, 3, 4);

        // Inputs: [5 steps, 3 features]
        NDArray xs = new ConcreteNDArray(new float[15], new Shape(5, 3));
        NDArray initH = new ConcreteNDArray(new float[4], new Shape(4));

        NDArray ys = gru.applySequence(xs, initH);
        assertEquals(new Shape(5, 4), ys.shape());

        // Test grad
        Function<PyTree, NDArray> lossFn = t -> {
            GRU m = Modules.unflatten(gru, t);
            return m.applySequence(xs, initH).sum();
        };

        PyTree params = Modules.flatten(gru);
        PyTree gradsTree = Grad.gradTree(lossFn).apply(params);
        GRU grads = Modules.unflatten(gru, gradsTree);

        assertNotNull(grads.wGates());
        assertNotNull(grads.uGates());
        assertNotNull(grads.bGates());
        assertNotNull(grads.wCandidate());
        assertNotNull(grads.uCandidate());
        assertNotNull(grads.bCandidate());
        assertEquals(new Shape(8, 3), grads.wGates().shape());
        assertEquals(new Shape(8, 4), grads.uGates().shape());
        assertEquals(new Shape(8), grads.bGates().shape());
        assertEquals(new Shape(4, 3), grads.wCandidate().shape());
        assertEquals(new Shape(4, 4), grads.uCandidate().shape());
        assertEquals(new Shape(4), grads.bCandidate().shape());
    }

    @Test
    public void testMultiHeadAttentionForwardAndGrad() {
        PRNGKey key = PRNGKey.key(12345);
        // MHA: embedDim=8, numHeads=2
        MultiHeadAttention mha = MultiHeadAttention.init(key, 8, 2);

        // Inputs: query [3, 8], key [4, 8], value [4, 8]
        NDArray query = new ConcreteNDArray(new float[24], new Shape(3, 8));
        NDArray keyArr = new ConcreteNDArray(new float[32], new Shape(4, 8));
        NDArray value = new ConcreteNDArray(new float[32], new Shape(4, 8));

        NDArray out = mha.apply(query, keyArr, value);
        assertEquals(new Shape(3, 8), out.shape());

        // Test batched inputs: [2, 3, 8], [2, 4, 8], [2, 4, 8]
        NDArray batchedQ = new ConcreteNDArray(new float[48], new Shape(2, 3, 8));
        NDArray batchedK = new ConcreteNDArray(new float[64], new Shape(2, 4, 8));
        NDArray batchedV = new ConcreteNDArray(new float[64], new Shape(2, 4, 8));

        NDArray batchedOut = mha.apply(batchedQ, batchedK, batchedV);
        assertEquals(new Shape(2, 3, 8), batchedOut.shape());

        // Test grad
        Function<PyTree, NDArray> lossFn = t -> {
            MultiHeadAttention m = Modules.unflatten(mha, t);
            return m.apply(query, keyArr, value).sum();
        };

        PyTree params = Modules.flatten(mha);
        PyTree gradsTree = Grad.gradTree(lossFn).apply(params);
        MultiHeadAttention grads = Modules.unflatten(mha, gradsTree);

        assertNotNull(grads.wQuery());
        assertNotNull(grads.wKey());
        assertNotNull(grads.wValue());
        assertNotNull(grads.wOut());
        assertEquals(new Shape(8, 8), grads.wQuery().shape());
        assertEquals(new Shape(8, 8), grads.wOut().shape());
    }
}
