package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;
import com.marmanis.jax4j.api.Lax;
import com.marmanis.jax4j.api.Numpy;

public record GRU(
    NDArray wGates,
    NDArray uGates,
    NDArray bGates,
    NDArray wCandidate,
    NDArray uCandidate,
    NDArray bCandidate,
    @Static int hiddenSize
) implements Module {

    private static NDArray scalar(int val) {
        return new ConcreteNDArray(new int[]{val}, new Shape());
    }

    private static NDArray scalar(float val) {
        return new ConcreteNDArray(new float[]{val}, new Shape());
    }

    public static GRU init(PRNGKey key, int inputSize, int hiddenSize) {
        PRNGKey[] keys = Random.split(key, 6);
        int gatesOut = 2 * hiddenSize;
        NDArray wG = Random.glorotUniform(keys[0], new Shape(gatesOut, inputSize), inputSize, gatesOut);
        NDArray uG = Random.glorotUniform(keys[1], new Shape(gatesOut, hiddenSize), hiddenSize, gatesOut);
        NDArray bG = new ConcreteNDArray(new float[gatesOut], new Shape(gatesOut));

        NDArray wC = Random.glorotUniform(keys[2], new Shape(hiddenSize, inputSize), inputSize, hiddenSize);
        NDArray uC = Random.glorotUniform(keys[3], new Shape(hiddenSize, hiddenSize), hiddenSize, hiddenSize);
        NDArray bC = new ConcreteNDArray(new float[hiddenSize], new Shape(hiddenSize));

        return new GRU(wG, uG, bG, wC, uC, bC, hiddenSize);
    }

    /**
     * Processes a single step: (h, x_t) -> [h_new, h_new].
     */
    public NDArray[] step(NDArray h, NDArray x) {
        NDArray xCol = x.reshape(new Shape(x.shape().dimensions()[0], 1));
        NDArray hCol = h.reshape(new Shape(hiddenSize, 1));
        NDArray gates = wGates.dot(xCol).reshape(new Shape(2 * hiddenSize))
            .add(uGates.dot(hCol).reshape(new Shape(2 * hiddenSize)))
            .add(bGates);
        NDArray gatesMat = gates.reshape(new Shape(2, hiddenSize));

        NDArray zGate = Numpy.take(gatesMat, scalar(0)).reshape(new Shape(hiddenSize)).sigmoid();
        NDArray rGate = Numpy.take(gatesMat, scalar(1)).reshape(new Shape(hiddenSize)).sigmoid();

        NDArray resetH = rGate.mul(h);
        NDArray resetHCol = resetH.reshape(new Shape(hiddenSize, 1));
        NDArray candidate = wCandidate.dot(xCol).reshape(new Shape(hiddenSize))
            .add(uCandidate.dot(resetHCol).reshape(new Shape(hiddenSize)))
            .add(bCandidate).tanh();

        NDArray oneMinusZ = scalar(1.0f).sub(zGate);
        NDArray hNew = oneMinusZ.mul(h).add(zGate.mul(candidate));

        return new NDArray[]{hNew, hNew};
    }

    /**
     * Processes a sequence of inputs of shape [seqLen, inputSize].
     * Returns the sequence of hidden states of shape [seqLen, hiddenSize].
     */
    public NDArray applySequence(NDArray xs, NDArray initH) {
        Lax.ScanResult scanRes = Lax.scan(this::step, initH, xs);
        return scanRes.ys();
    }
}
