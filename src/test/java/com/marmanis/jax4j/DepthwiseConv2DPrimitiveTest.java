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
 * Sanity checks for the {@link com.marmanis.jax4j.ir.Primitive#DEPTHWISE_CONV_2D}
 * primitive: shape correctness and numerical gradient check w.r.t. the kernel.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class DepthwiseConv2DPrimitiveTest {

    @Test
    public void forward_shape_correctness() {
        Random rng = new Random(0);
        float[] xd = new float[1 * 5 * 5 * 3];
        for (int i = 0; i < xd.length; i++) xd[i] = rng.nextFloat();
        NDArray x = new ConcreteNDArray(xd, new Shape(1, 5, 5, 3));

        int depthMult = 2;
        float[] kd = new float[3 * 3 * 3 * depthMult];
        for (int i = 0; i < kd.length; i++) kd[i] = (rng.nextFloat() - 0.5f) * 0.1f;
        NDArray k = new ConcreteNDArray(kd, new Shape(3, 3, 3, depthMult));

        NDArray y = Convolution.depthwiseConv2dEager(x, k, new int[]{1, 1}, "valid", depthMult);
        assertEquals(new Shape(1, 3, 3, 6), y.shape());
    }

    @Test
    public void numerical_gradient_check_wrt_kernel() {
        // 1 batch, 3x3 input, 2 channels, depthMult=2 (output has 4 channels).
        Random rng = new Random(42);
        float[] xd = new float[1 * 3 * 3 * 2];
        for (int i = 0; i < xd.length; i++) xd[i] = (rng.nextFloat() - 0.5f);
        NDArray x = new ConcreteNDArray(xd, new Shape(1, 3, 3, 2));

        int depthMult = 2;
        float[] kd = new float[2 * 2 * 2 * depthMult];
        for (int i = 0; i < kd.length; i++) kd[i] = (rng.nextFloat() - 0.5f);
        Shape kShape = new Shape(2, 2, 2, depthMult);
        NDArray k0 = new ConcreteNDArray(kd, kShape);

        Map<String, PyTree> paramMap = new LinkedHashMap<>();
        paramMap.put("k", PyTree.leaf(k0));
        PyTree params = PyTree.dict(paramMap);

        java.util.function.Function<PyTree, NDArray> loss = pt -> {
            NDArray kk = ((PyTree.Leaf) ((PyTree.MapNode) pt).children().get("k")).value();
            NDArray y = Convolution.depthwiseConv2dForward(x, kk, new int[]{1, 1}, "valid", depthMult);
            return y.mul(y).sum();
        };

        PyTree grads = Grad.gradTree(loss).apply(params);
        NDArray dK = ((PyTree.Leaf) ((PyTree.MapNode) grads).children().get("k")).value();
        float[] analytic = dK.toFloatArray();

        float eps = 1e-3f;
        for (int i = 0; i < kd.length; i++) {
            float[] kPlus = kd.clone(); kPlus[i] += eps;
            float[] kMinus = kd.clone(); kMinus[i] -= eps;
            float lp = lossValue(x, new ConcreteNDArray(kPlus, kShape), depthMult);
            float lm = lossValue(x, new ConcreteNDArray(kMinus, kShape), depthMult);
            float num = (lp - lm) / (2f * eps);
            assertEquals(num, analytic[i], 1e-2f, "kernel[" + i + "] grad mismatch");
        }
    }

    private static float lossValue(NDArray x, NDArray k, int depthMult) {
        NDArray y = Convolution.depthwiseConv2dEager(x, k, new int[]{1, 1}, "valid", depthMult);
        float[] yd = y.toFloatArray();
        float sum = 0f;
        for (float v : yd) sum += v * v;
        return sum;
    }
}
