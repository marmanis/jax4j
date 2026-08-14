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

public record LSTM(
    NDArray wGates,
    NDArray uGates,
    NDArray bGates,
    @Static int hiddenSize
) implements Module {

    private static NDArray scalar(int val) {
        return new ConcreteNDArray(new int[]{val}, new Shape());
    }

    public static LSTM init(PRNGKey key, int inputSize, int hiddenSize) {
        PRNGKey[] keys = Random.split(key, 3);
        int outFeatures = 4 * hiddenSize;
        NDArray wG = Random.glorotUniform(keys[0], new Shape(outFeatures, inputSize), inputSize, outFeatures);
        NDArray uG = Random.glorotUniform(keys[1], new Shape(outFeatures, hiddenSize), hiddenSize, outFeatures);
        NDArray bG = new ConcreteNDArray(new float[outFeatures], new Shape(outFeatures));
        return new LSTM(wG, uG, bG, hiddenSize);
    }

    /**
     * Processes a single step: (carry, x_t) -> [newCarry, h_new].
     * carry has shape [2 * hiddenSize], representing concatenated h and c.
     */
    public NDArray[] step(NDArray carry, NDArray x) {
        NDArray carryMat = carry.reshape(new Shape(2, hiddenSize));
        NDArray h = Numpy.take(carryMat, scalar(0)).reshape(new Shape(hiddenSize));
        NDArray c = Numpy.take(carryMat, scalar(1)).reshape(new Shape(hiddenSize));

        NDArray xCol = x.reshape(new Shape(x.shape().dimensions()[0], 1));
        NDArray hCol = h.reshape(new Shape(hiddenSize, 1));
        NDArray gates = wGates.dot(xCol).reshape(new Shape(4 * hiddenSize))
            .add(uGates.dot(hCol).reshape(new Shape(4 * hiddenSize)))
            .add(bGates);
        NDArray gatesMat = gates.reshape(new Shape(4, hiddenSize));

        NDArray iGate = Numpy.take(gatesMat, scalar(0)).reshape(new Shape(hiddenSize)).sigmoid();
        NDArray fGate = Numpy.take(gatesMat, scalar(1)).reshape(new Shape(hiddenSize)).sigmoid();
        NDArray gGate = Numpy.take(gatesMat, scalar(2)).reshape(new Shape(hiddenSize)).tanh();
        NDArray oGate = Numpy.take(gatesMat, scalar(3)).reshape(new Shape(hiddenSize)).sigmoid();

        NDArray cNew = fGate.mul(c).add(iGate.mul(gGate));
        NDArray hNew = oGate.mul(cNew.tanh());

        NDArray carryNew = Numpy.concatenate(java.util.List.of(hNew, cNew), 0);

        return new NDArray[]{carryNew, hNew};
    }

    /**
     * Processes a sequence of inputs of shape [seqLen, inputSize].
     * Returns the sequence of hidden states of shape [seqLen, hiddenSize].
     */
    public NDArray applySequence(NDArray xs, NDArray initH, NDArray initC) {
        NDArray initCarry = Numpy.concatenate(java.util.List.of(initH, initC), 0);
        Lax.ScanResult scanRes = Lax.scan(this::step, initCarry, xs);
        return scanRes.ys();
    }
}
