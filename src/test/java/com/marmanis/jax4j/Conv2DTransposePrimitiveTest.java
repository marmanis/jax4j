package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Convolution;
import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Sanity checks for the {@link com.marmanis.jax4j.ir.Primitive#CONV_2D_TRANSPOSE}
 * primitive: shape correctness for stride 1 and stride 2, plus numerical
 * gradient checks with respect to both the kernel and one input pixel.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class Conv2DTransposePrimitiveTest {

    @Test
    public void forward_shape_stride1() {
        NDArray x = filled(new Shape(1, 3, 3, 2), 0);
        NDArray k = filled(new Shape(3, 3, 4, 2), 1);
        NDArray y = Convolution.conv2dTransposeEager(x, k, new int[]{1, 1}, "valid");
        assertEquals(new Shape(1, 5, 5, 4), y.shape());
    }

    @Test
    public void forward_shape_stride2() {
        NDArray x = filled(new Shape(1, 3, 3, 2), 0);
        NDArray k = filled(new Shape(3, 3, 4, 2), 1);
        NDArray y = Convolution.conv2dTransposeEager(x, k, new int[]{2, 2}, "valid");
        assertEquals(new Shape(1, 7, 7, 4), y.shape());
    }

    @Test
    public void numerical_gradient_wrt_kernel() {
        Random rng = new Random(11);
        Shape xShape = new Shape(1, 3, 3, 2);
        Shape kShape = new Shape(2, 2, 3, 2);
        float[] xd = rand(rng, (int) xShape.size());
        float[] kd = rand(rng, (int) kShape.size());
        NDArray x = new ConcreteNDArray(xd, xShape);

        Map<String, PyTree> paramMap = new LinkedHashMap<>();
        paramMap.put("k", PyTree.leaf(new ConcreteNDArray(kd, kShape)));
        PyTree params = PyTree.dict(paramMap);

        java.util.function.Function<PyTree, NDArray> loss = pt -> {
            NDArray kk = ((PyTree.Leaf) ((PyTree.MapNode) pt).children().get("k")).value();
            NDArray y = Convolution.conv2dTransposeForward(x, kk, new int[]{2, 2}, "valid");
            return y.mul(y).sum();
        };

        PyTree grads = Grad.gradTree(loss).apply(params);
        float[] analytic = ((PyTree.Leaf) ((PyTree.MapNode) grads).children().get("k"))
                .value().toFloatArray();

        float eps = 1e-3f;
        for (int i = 0; i < kd.length; i++) {
            float[] kp = kd.clone(); kp[i] += eps;
            float[] km = kd.clone(); km[i] -= eps;
            float lp = kernelLoss(x, new ConcreteNDArray(kp, kShape));
            float lm = kernelLoss(x, new ConcreteNDArray(km, kShape));
            float num = (lp - lm) / (2f * eps);
            assertEquals(num, analytic[i], 1e-2f, "kernel[" + i + "] grad mismatch");
        }
    }

    @Test
    public void numerical_gradient_wrt_input() {
        Random rng = new Random(23);
        Shape xShape = new Shape(1, 3, 3, 2);
        Shape kShape = new Shape(2, 2, 3, 2);
        float[] xd = rand(rng, (int) xShape.size());
        float[] kd = rand(rng, (int) kShape.size());
        NDArray k = new ConcreteNDArray(kd, kShape);

        Map<String, PyTree> paramMap = new LinkedHashMap<>();
        paramMap.put("x", PyTree.leaf(new ConcreteNDArray(xd, xShape)));
        PyTree params = PyTree.dict(paramMap);

        java.util.function.Function<PyTree, NDArray> loss = pt -> {
            NDArray xx = ((PyTree.Leaf) ((PyTree.MapNode) pt).children().get("x")).value();
            NDArray y = Convolution.conv2dTransposeForward(xx, k, new int[]{2, 2}, "valid");
            return y.mul(y).sum();
        };

        PyTree grads = Grad.gradTree(loss).apply(params);
        float[] analytic = ((PyTree.Leaf) ((PyTree.MapNode) grads).children().get("x"))
                .value().toFloatArray();

        float eps = 1e-3f;
        int i = 5;   // one input pixel
        float[] xp = xd.clone(); xp[i] += eps;
        float[] xm = xd.clone(); xm[i] -= eps;
        float lp = kernelLoss(new ConcreteNDArray(xp, xShape), k);
        float lm = kernelLoss(new ConcreteNDArray(xm, xShape), k);
        float num = (lp - lm) / (2f * eps);
        assertEquals(num, analytic[i], 1e-2f, "input[" + i + "] grad mismatch");
    }

    private static float kernelLoss(NDArray x, NDArray k) {
        float[] y = Convolution.conv2dTransposeEager(x, k, new int[]{2, 2}, "valid").toFloatArray();
        float s = 0f;
        for (float v : y) s += v * v;
        return s;
    }

    private static NDArray filled(Shape s, long seed) {
        Random rng = new Random(seed);
        float[] d = new float[(int) s.size()];
        for (int i = 0; i < d.length; i++) d[i] = rng.nextFloat();
        return new ConcreteNDArray(d, s);
    }

    private static float[] rand(Random rng, int n) {
        float[] d = new float[n];
        for (int i = 0; i < n; i++) d[i] = (rng.nextFloat() - 0.5f);
        return d;
    }
}
