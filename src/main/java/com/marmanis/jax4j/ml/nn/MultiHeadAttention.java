package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;
import com.marmanis.jax4j.api.Vmap;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.util.List;
import java.util.function.Function;

public record MultiHeadAttention(
    NDArray wQuery, NDArray bQuery,
    NDArray wKey, NDArray bKey,
    NDArray wValue, NDArray bValue,
    NDArray wOut, NDArray bOut,
    @Static int numHeads,
    @Static int headDim
) implements Module {

    private static NDArray scalar(float val) {
        return new ConcreteNDArray(new float[]{val}, new Shape());
    }

    public static MultiHeadAttention init(PRNGKey key, int embedDim, int numHeads) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        int headDim = embedDim / numHeads;
        PRNGKey[] keys = Random.split(key, 4);

        NDArray wQ = Random.glorotUniform(keys[0], new Shape(embedDim, embedDim), embedDim, embedDim);
        NDArray bQ = new ConcreteNDArray(new float[embedDim], new Shape(embedDim));

        NDArray wK = Random.glorotUniform(keys[1], new Shape(embedDim, embedDim), embedDim, embedDim);
        NDArray bK = new ConcreteNDArray(new float[embedDim], new Shape(embedDim));

        NDArray wV = Random.glorotUniform(keys[2], new Shape(embedDim, embedDim), embedDim, embedDim);
        NDArray bV = new ConcreteNDArray(new float[embedDim], new Shape(embedDim));

        NDArray wO = Random.glorotUniform(keys[3], new Shape(embedDim, embedDim), embedDim, embedDim);
        NDArray bO = new ConcreteNDArray(new float[embedDim], new Shape(embedDim));

        return new MultiHeadAttention(wQ, bQ, wK, bK, wV, bV, wO, bO, numHeads, headDim);
    }

    public NDArray apply(NDArray query, NDArray key, NDArray value) {
        Shape qShape = query.shape();
        if (qShape.rank() == 3) {
            PyTree inputs = PyTree.list(PyTree.leaf(query), PyTree.leaf(key), PyTree.leaf(value));
            Function<PyTree, PyTree> batched = Vmap.vmapTree(t -> {
                List<NDArray> leaves = PyTrees.flatten(t);
                NDArray out = this.apply(leaves.get(0), leaves.get(1), leaves.get(2));
                return PyTree.list(PyTree.leaf(out));
            }, new int[]{0, 0, 0}, new int[]{0});
            return PyTrees.flatten(batched.apply(inputs)).get(0);
        }
        if (qShape.rank() != 2) {
            throw new IllegalArgumentException("MultiHeadAttention input must be of rank 2 [seqLen, embedDim] or rank 3 [batch, seqLen, embedDim]");
        }

        int seqLenQ = qShape.dimensions()[0];
        int seqLenK = key.shape().dimensions()[0];
        int embedDim = qShape.dimensions()[1];

        NDArray qProj = query.dot(wQuery).add(bQuery);
        NDArray kProj = key.dot(wKey).add(bKey);
        NDArray vProj = value.dot(wValue).add(bValue);

        NDArray qHeads = qProj.reshape(new Shape(seqLenQ, numHeads, headDim)).transpose(1, 0, 2);
        NDArray kHeads = kProj.reshape(new Shape(seqLenK, numHeads, headDim)).transpose(1, 0, 2);
        NDArray vHeads = vProj.reshape(new Shape(seqLenK, numHeads, headDim)).transpose(1, 0, 2);

        PyTree args = PyTree.list(PyTree.leaf(qHeads), PyTree.leaf(kHeads), PyTree.leaf(vHeads));
        Function<PyTree, PyTree> batchedAttention = Vmap.vmapTree(t -> {
            List<NDArray> leaves = PyTrees.flatten(t);
            NDArray q = leaves.get(0); // [seqLenQ, headDim]
            NDArray k = leaves.get(1); // [seqLenK, headDim]
            NDArray v = leaves.get(2); // [seqLenK, headDim]

            NDArray scores = q.dot(k.transpose(1, 0)).div(scalar((float) Math.sqrt(headDim)));
            NDArray weights = com.marmanis.jax4j.api.Nn.softmax(scores);
            NDArray headOut = weights.dot(v); // [seqLenQ, headDim]
            return PyTree.list(PyTree.leaf(headOut));
        }, new int[]{0, 0, 0}, new int[]{0});

        PyTree resTree = batchedAttention.apply(args);
        NDArray attnOutHeads = PyTrees.flatten(resTree).get(0); // [numHeads, seqLenQ, headDim]

        NDArray attnOut = attnOutHeads.transpose(1, 0, 2).reshape(new Shape(seqLenQ, embedDim));

        return attnOut.dot(wOut).add(bOut);
    }
}
