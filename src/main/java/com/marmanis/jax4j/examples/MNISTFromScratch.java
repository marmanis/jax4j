package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Nn;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Port of JAX's "MNIST classifier from scratch" example
 * (jax-ml/jax examples/mnist_classifier_fromscratch.py) to jax4j.
 *
 * <p>jax4j has no {@code argmax}, so {@code accuracy()} is not differentiated
 * and drops out of the traced API to work directly on {@code float[]}.
 * Everything else uses core differentiable primitives: {@code tanh} directly,
 * the log-softmax in {@link #predict} via {@link Nn#softmax} (built on
 * {@code NDArray.sum(axis, keepDims)}), and the loss's
 * {@code sum(preds*targets, axis=-1)} via the same axis-aware reduction.
 *
 * <p>Training data is the real MNIST dataset, fetched and cached by
 * {@link DownloadManager#mnist(boolean)}.
 */
public class MNISTFromScratch {

    private static final int INPUT_DIM = 784; // fixed by the dataset: MNIST images are 28x28
    private static final int NUM_CLASSES = 10;
    // Smaller than the original {784, 1024, 1024, 10}: this is a pure-Java,
    // host-only forward/backward pass with no TornadoVM kernels for
    // dot/exp/log, so a smaller hidden width keeps the demo running in a
    // reasonable time while still exercising a multi-layer tanh network.
    private static final int[] LAYER_SIZES = {INPUT_DIM, 1024, 1024, NUM_CLASSES};
    private static final float PARAM_SCALE = 0.1f;
    private static final float STEP_SIZE = 0.1f;
    private static final int NUM_EPOCHS = 10;
    private static final int BATCH_SIZE = 128;

    public static void main(String[] args) throws IOException {
        ExampleBackend.selectFromArgs(args);

        PRNGKey[] topKeys = Random.split(PRNGKey.key(0));
        PRNGKey paramKey = topKeys[0];
        PRNGKey loopKey = topKeys[1];

        DownloadManager.MnistData data = DownloadManager.mnist(true);
        Dataset train = new Dataset(data.trainImages(), data.trainLabels(), data.trainImages().shape().dimensions()[0]);
        Dataset test = new Dataset(data.testImages(), data.testLabels(), data.testImages().shape().dimensions()[0]);

        PyTree params = initRandomParams(PARAM_SCALE, LAYER_SIZES, paramKey);

        int numBatches = Math.max(1, train.size / BATCH_SIZE);
        System.out.println("Starting MNIST-from-scratch training (real MNIST data)...");
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            long start = System.nanoTime();
            PRNGKey[] epochKeys = Random.split(loopKey);
            loopKey = epochKeys[0];
            int[] perm = shuffledIndices(train.size, epochKeys[1]);
            for (int b = 0; b < numBatches; b++) {
                int[] batchIdx = slice(perm, b * BATCH_SIZE, BATCH_SIZE);
                NDArray inputs = gather(train.inputs, INPUT_DIM, batchIdx);
                NDArray targets = gather(train.targets, NUM_CLASSES, batchIdx);

                Function<PyTree, NDArray> lossFn = p -> loss(p, inputs, targets);
                PyTree grads = JAX.gradTree(lossFn).apply(params);
                params = PyTrees.map2(
                    (p, g) -> p.sub(g.mul(scalar(STEP_SIZE, g))),
                    params, grads);
            }
            double epochSec = (System.nanoTime() - start) / 1e9;

            double trainAcc = accuracy(params, train);
            double testAcc = accuracy(params, test);
            System.out.printf("Epoch %d in %.2f sec -- train acc: %.4f, test acc: %.4f%n",
                epoch, epochSec, trainAcc, testAcc);
        }
    }

    // ---- model ----

    private static PyTree initRandomParams(float scale, int[] layerSizes, PRNGKey key) {
        List<PyTree> layers = new ArrayList<>();
        PRNGKey layerKey = key;
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int m = layerSizes[i];
            int n = layerSizes[i + 1];
            PRNGKey[] keys = Random.split(layerKey, 3);
            layerKey = keys[0];
            NDArray w = randomGaussian(scale, new Shape(m, n), keys[1]);
            NDArray b = randomGaussian(scale, new Shape(n), keys[2]);
            layers.add(PyTree.list(PyTree.leaf(w), PyTree.leaf(b)));
        }
        return PyTree.list(layers.toArray(new PyTree[0]));
    }

    private static NDArray predict(PyTree params, NDArray inputs) {
        List<PyTree> layers = ((PyTree.ListNode) params).children();
        NDArray activations = inputs;
        for (int i = 0; i < layers.size() - 1; i++) {
            List<PyTree> layer = ((PyTree.ListNode) layers.get(i)).children();
            NDArray w = leaf(layer.get(0));
            NDArray b = leaf(layer.get(1));
            NDArray outputs = activations.dot(w).add(b);
            activations = outputs.tanh();
        }
        List<PyTree> finalLayer = ((PyTree.ListNode) layers.get(layers.size() - 1)).children();
        NDArray finalW = leaf(finalLayer.get(0));
        NDArray finalB = leaf(finalLayer.get(1));
        NDArray logits = activations.dot(finalW).add(finalB);
        return Nn.softmax(logits).log();
    }

    private static NDArray loss(PyTree params, NDArray inputs, NDArray targets) {
        NDArray preds = predict(params, inputs);
        NDArray rowSums = preds.mul(targets).sum(-1);
        NDArray negOne = new ConcreteNDArray(new float[]{-1f}, new Shape(1));
        return rowSums.mean().mul(negOne);
    }

    private static double accuracy(PyTree params, Dataset data) {
        NDArray preds = predict(params, data.inputs);
        // Real INT32 label indices and a real BOOL match mask -- no float-hacking.
        NDArray predClasses = preds.argmax(-1);
        NDArray targetClasses = data.targets.argmax(-1);
        boolean[] matches = predClasses.eq(targetClasses).toBoolArray();
        int correct = 0;
        for (boolean m : matches) if (m) correct++;
        return (double) correct / data.size;
    }

    // ---- helpers built from existing differentiable primitives ----

    private static NDArray scalar(float v, NDArray like) {
        float[] d = new float[like.toFloatArray().length];
        java.util.Arrays.fill(d, v);
        return new ConcreteNDArray(d, like.shape());
    }

    private static NDArray leaf(PyTree t) {
        return ((PyTree.Leaf) t).value();
    }

    private static NDArray randomGaussian(float scale, Shape shape, PRNGKey key) {
        NDArray scaleArr = new ConcreteNDArray(new float[]{scale}, new Shape(1));
        return Random.normal(key, shape).mul(scaleArr);
    }

    // ---- dataset wrapper ----

    private record Dataset(NDArray inputs, NDArray targets, int size) {}

    private static int[] shuffledIndices(int n, PRNGKey key) {
        return Random.permutation(key, n).toIntArray();
    }

    private static int[] slice(int[] arr, int from, int length) {
        int end = Math.min(from + length, arr.length);
        if (from >= end) {
            return new int[]{arr[0]};
        }
        int[] result = new int[end - from];
        System.arraycopy(arr, from, result, 0, end - from);
        return result;
    }

    private static NDArray gather(NDArray data, int rowWidth, int[] rowIndices) {
        float[] src = data.toFloatArray();
        float[] dst = new float[rowIndices.length * rowWidth];
        for (int i = 0; i < rowIndices.length; i++) {
            System.arraycopy(src, rowIndices[i] * rowWidth, dst, i * rowWidth, rowWidth);
        }
        return new ConcreteNDArray(dst, new Shape(rowIndices.length, rowWidth));
    }
}
