package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.data.DataLoader;
import com.marmanis.jax4j.data.Dataset;
import com.marmanis.jax4j.data.Generators;
import com.marmanis.jax4j.data.Mnist;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the com.marmanis.jax4j.data package.
 */
public class DataTest {

    @Test
    public void testDataLoaderCollationAndShuffling() {
        // Create a dummy dataset of 5 examples
        Dataset dummyDataset = new Dataset() {
            @Override
            public int size() {
                return 5;
            }

            @Override
            public PyTree get(int index) {
                // Each item contains x: [3] and y: [1]
                NDArray x = new ConcreteNDArray(new float[]{index, index * 2f, index * 3f}, new Shape(3));
                NDArray y = new ConcreteNDArray(new float[]{index}, new Shape(1));
                return PyTree.list(PyTree.leaf(x), PyTree.leaf(y));
            }
        };

        // 1. Test batching without shuffling
        DataLoader loader = new DataLoader(dummyDataset, 2, false, null);
        Iterator<PyTree> it = loader.iterator();

        assertTrue(it.hasNext());
        PyTree batch1 = it.next();
        List<NDArray> leaves1 = PyTrees.flatten(batch1);
        assertEquals(2, leaves1.size());
        assertEquals(new Shape(2, 3), leaves1.get(0).shape());
        assertEquals(new Shape(2, 1), leaves1.get(1).shape());

        // Check values of Batch 1: indices [0, 1]
        assertArrayEquals(new float[]{0f, 0f, 0f, 1f, 2f, 3f}, leaves1.get(0).toFloatArray());
        assertArrayEquals(new float[]{0f, 1f}, leaves1.get(1).toFloatArray());

        assertTrue(it.hasNext());
        PyTree batch2 = it.next();
        List<NDArray> leaves2 = PyTrees.flatten(batch2);
        assertEquals(new Shape(2, 3), leaves2.get(0).shape());
        assertEquals(new Shape(2, 1), leaves2.get(1).shape());
        // Batch 2: indices [2, 3]
        assertArrayEquals(new float[]{2f, 4f, 6f, 3f, 6f, 9f}, leaves2.get(0).toFloatArray());
        assertArrayEquals(new float[]{2f, 3f}, leaves2.get(1).toFloatArray());

        assertTrue(it.hasNext());
        PyTree batch3 = it.next();
        List<NDArray> leaves3 = PyTrees.flatten(batch3);
        // Batch 3: index [4] (remainder batch)
        assertEquals(new Shape(1, 3), leaves3.get(0).shape());
        assertEquals(new Shape(1, 1), leaves3.get(1).shape());
        assertArrayEquals(new float[]{4f, 8f, 12f}, leaves3.get(0).toFloatArray());
        assertArrayEquals(new float[]{4f}, leaves3.get(1).toFloatArray());

        assertTrue(!it.hasNext());

        // 2. Test shuffling using PRNGKey
        PRNGKey key = PRNGKey.key(42);
        DataLoader shuffledLoader = new DataLoader(dummyDataset, 5, true, key);
        PyTree shuffledBatch = shuffledLoader.iterator().next();
        List<NDArray> shuffledLeaves = PyTrees.flatten(shuffledBatch);
        assertEquals(new Shape(5, 3), shuffledLeaves.get(0).shape());
        assertEquals(new Shape(5, 1), shuffledLeaves.get(1).shape());

        // Verify that shuffling actually changed the sequence from [0, 1, 2, 3, 4]
        float[] shuffledLabels = shuffledLeaves.get(1).toFloatArray();
        boolean isSequential = true;
        for (int i = 0; i < 5; i++) {
            if (shuffledLabels[i] != i) {
                isSequential = false;
                break;
            }
        }
        assertTrue(!isSequential, "Shuffled batch labels should not match sequential order");
    }

    @Test
    public void testGenerators() {
        PRNGKey key = PRNGKey.key(1234);

        // Regression
        Generators.RegressionResult regression = Generators.makeRegression(key, 50, 5, 2, 0.1f);
        assertEquals(new Shape(50, 5), regression.xs().shape());
        assertEquals(new Shape(50, 2), regression.ys().shape());
        assertEquals(new Shape(5, 2), regression.coef().shape());
        assertEquals(new Shape(2), regression.intercept().shape());

        // Moons
        Generators.ClassificationResult moons = Generators.makeMoons(key, 60, 0.05f);
        assertEquals(new Shape(60, 2), moons.xs().shape());
        assertEquals(new Shape(60), moons.ys().shape());

        // Circles
        Generators.ClassificationResult circles = Generators.makeCircles(key, 80, 0.05f, 0.8f);
        assertEquals(new Shape(80, 2), circles.xs().shape());
        assertEquals(new Shape(80), circles.ys().shape());
    }

    @Test
    public void testMnistParsing() throws IOException {
        // Create mock binary bytes for MNIST
        // Images: Magic 2051, Count 2, Rows 2, Cols 2 (total size 4 + 4 + 4 + 4 + 2*2*2 = 24 bytes)
        ByteBuffer imgBuf = ByteBuffer.allocate(24);
        imgBuf.putInt(2051); // magic
        imgBuf.putInt(2);    // count
        imgBuf.putInt(2);    // rows
        imgBuf.putInt(2);    // cols
        imgBuf.put(new byte[]{0, 127, (byte) 255, 0, 100, (byte) 200, 50, 10});

        // Labels: Magic 2049, Count 2 (total size 4 + 4 + 2 = 10 bytes)
        ByteBuffer lblBuf = ByteBuffer.allocate(10);
        lblBuf.putInt(2049); // magic
        lblBuf.putInt(2);    // count
        lblBuf.put(new byte[]{5, 3});

        Mnist.MnistDataset dataset = Mnist.parse(imgBuf.array(), lblBuf.array());
        assertEquals(2, dataset.size());

        NDArray xs = dataset.getXs();
        NDArray ys = dataset.getYs();

        assertEquals(new Shape(2, 4), xs.shape());
        assertEquals(new Shape(2), ys.shape());

        assertArrayEquals(new float[]{5f, 3f}, ys.toFloatArray());
        // Check pixel values (normalized)
        float[] pixels = xs.toFloatArray();
        assertEquals(0.0f, pixels[0], 1e-4f);
        assertEquals(127f / 255.0f, pixels[1], 1e-4f);
        assertEquals(1.0f, pixels[2], 1e-4f);

        // Check PyTree get retrieval
        PyTree firstItem = dataset.get(0);
        List<NDArray> itemLeaves = PyTrees.flatten(firstItem);
        assertEquals(new Shape(4), itemLeaves.get(0).shape());
        assertEquals(new Shape(), itemLeaves.get(1).shape());
        assertEquals(5f, itemLeaves.get(1).toFloatArray()[0]);
    }
}
