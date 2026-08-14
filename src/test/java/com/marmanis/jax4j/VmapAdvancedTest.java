package com.marmanis.jax4j;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Vmap;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

import static com.marmanis.jax4j.testutil.GradChecker.scalar;
import static com.marmanis.jax4j.testutil.GradChecker.vector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Advanced tests for JAX.vmap with custom axes, PyTree vmaps, and Jacobians.
 */
public class VmapAdvancedTest {

    @Test
    public void testVmapWithCustomAxes() {
        // batchedArg shape: [2, 3, 4]
        // We set batch dimension at axis 1 (size 3), and example shape is [2, 4].
        float[] data = new float[2 * 3 * 4];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i;
        }
        NDArray batchedArg = new ConcreteNDArray(data, new Shape(2, 3, 4));

        Function<NDArray, NDArray> fn = x -> x.mul(scalar(2.0f));

        // Map along input axis 1, place output batch along axis 1
        NDArray result = Vmap.vmap(fn, 1, 1).apply(batchedArg);

        assertEquals(new Shape(2, 3, 4), result.shape());
        float[] resData = result.toFloatArray();
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i] * 2.0f, resData[i], 1e-4f);
        }
    }

    @Test
    public void testVmapTree() {
        NDArray a = vector(1.0f, 2.0f, 3.0f); // batch size 3 on axis 0
        NDArray b = new ConcreteNDArray(new float[]{10, 20, 30, 40, 50, 60}, new Shape(2, 3)); // batch size 3 on axis 1

        PyTree tree = PyTree.list(PyTree.leaf(a), PyTree.leaf(b));

        // fn returns a list of two leaves: a * 2, and b + 5
        Function<PyTree, PyTree> fn = t -> {
            List<NDArray> leaves = PyTrees.flatten(t);
            NDArray leafA = leaves.get(0);
            NDArray leafB = leaves.get(1);
            return PyTree.list(
                PyTree.leaf(leafA.mul(scalar(2.0f))),
                PyTree.leaf(leafB.add(scalar(5.0f)))
            );
        };

        // inAxes: batch axis 0 for leaf 0, batch axis 1 for leaf 1
        // outAxes: batch axis 0 for output leaf 0, batch axis 1 for output leaf 1
        int[] inAxes = {0, 1};
        int[] outAxes = {0, 1};

        PyTree batchedResult = Vmap.vmapTree(fn, inAxes, outAxes).apply(tree);
        List<NDArray> resLeaves = PyTrees.flatten(batchedResult);

        assertEquals(2, resLeaves.size());
        assertArrayEquals(new float[]{2.0f, 4.0f, 6.0f}, resLeaves.get(0).toFloatArray(), 1e-4f);
        assertArrayEquals(new float[]{15, 25, 35, 45, 55, 65}, resLeaves.get(1).toFloatArray(), 1e-4f);
    }

    @Test
    public void testJacobianForwardAndReverse() {
        // f(x) = [sin(x_0) * cos(x_1), x_0^2 + x_1^3]
        Function<NDArray, NDArray> fn = x -> {
            float[] data = x.toFloatArray();
            NDArray x0 = scalar(data[0]);
            NDArray x1 = scalar(data[1]);
            NDArray y0 = x0.sin().mul(x1.cos());
            NDArray y1 = x0.mul(x0).add(x1.mul(x1).mul(x1));
            // Concatenate them to [2]
            return JAX.make_jaxpr(t -> y0, x0).equations().isEmpty() 
                ? vector(y0.toFloatArray()[0], y1.toFloatArray()[0])
                : y0.add(scalar(0.0f)); // dummy for tracing shape
        };

        // Since fn must be differentiable, let's write it purely using NDArray primitives so tracing works!
        // Input: [2] -> Output: [2]
        Function<NDArray, NDArray> differentiableFn = x -> {
            NDArray mask0 = new ConcreteNDArray(new float[]{1.0f, 0.0f}, new Shape(2), DType.FLOAT32, x.device());
            NDArray mask1 = new ConcreteNDArray(new float[]{0.0f, 1.0f}, new Shape(2), DType.FLOAT32, x.device());
            NDArray x0 = x.mul(mask0).sum().reshape(new Shape(1));
            NDArray x1 = x.mul(mask1).sum().reshape(new Shape(1));
            NDArray y0 = x0.sin().mul(x1.cos());
            NDArray y1 = x0.mul(x0).add(x1.mul(x1).mul(x1));
            return com.marmanis.jax4j.api.Numpy.concatenate(List.of(y0, y1), 0);
        };

        NDArray arg = vector(0.5f, 0.8f);

        // Compute forward and reverse Jacobians
        NDArray jacFwd = JAX.jacfwd(differentiableFn, arg);
        NDArray jacRev = JAX.jacrev(differentiableFn, arg);

        assertEquals(new Shape(2, 2), jacFwd.shape());
        assertEquals(new Shape(2, 2), jacRev.shape());

        // Analytical Jacobians at (0.5, 0.8):
        // dy0/dx0 = cos(0.5) * cos(0.8) = 0.8775825 * 0.6967067 = 0.611417
        // dy0/dx1 = -sin(0.5) * sin(0.8) = -0.4794255 * 0.7173561 = -0.343918
        // dy1/dx0 = 2 * 0.5 = 1.0
        // dy1/dx1 = 3 * 0.8^2 = 1.92
        float[] expected = {
            0.611417f, -0.343918f,
            1.0f, 1.92f
        };

        assertArrayEquals(expected, jacFwd.toFloatArray(), 1e-4f);
        assertArrayEquals(expected, jacRev.toFloatArray(), 1e-4f);
    }
}
