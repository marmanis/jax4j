package com.marmanis.jax4j;

import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.GridSampling;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Sanity checks for the {@link com.marmanis.jax4j.ir.Primitive#GRID_SAMPLE_2D}
 * primitive: identity grid, known 2x upsampling, padding modes, and a
 * numerical gradient check w.r.t. the input image.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class GridSample2DPrimitiveTest {

    @Test
    public void identity_grid_is_pass_through() {
        int B = 1, H = 3, W = 3, C = 2;
        float[] xd = new float[B * H * W * C];
        for (int i = 0; i < xd.length; i++) xd[i] = i;
        NDArray x = new ConcreteNDArray(xd, new Shape(B, H, W, C));

        float[] gd = new float[B * H * W * 2];
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {
                int base = (oy * W + ox) * 2;
                gd[base] = oy;
                gd[base + 1] = ox;
            }
        }
        NDArray g = new ConcreteNDArray(gd, new Shape(B, H, W, 2));

        NDArray y = GridSampling.gridSample2dForward(x, g, "zeros");
        assertArrayEquals(xd, y.toFloatArray(), 1e-5f);
    }

    @Test
    public void two_by_two_upsampling_bilinear() {
        // Input 2x2, single channel:
        //   1  3
        //   5  7
        float[] xd = new float[]{1, 3, 5, 7};
        NDArray x = new ConcreteNDArray(xd, new Shape(1, 2, 2, 1));

        // Output 4x4 with border padding, mapping (oy, ox) -> (oy/2, ox/2).
        int outH = 4, outW = 4;
        float[] gd = new float[outH * outW * 2];
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                int base = (oy * outW + ox) * 2;
                gd[base] = oy / 2.0f;
                gd[base + 1] = ox / 2.0f;
            }
        }
        NDArray g = new ConcreteNDArray(gd, new Shape(1, outH, outW, 2));
        NDArray y = GridSampling.gridSample2dForward(x, g, "border");
        float[] yd = y.toFloatArray();

        // Spot-check a few pixels.
        // (0,0)->(0,0)      = 1
        // (0,1)->(0,0.5)    = 0.5*1 + 0.5*3 = 2
        // (1,0)->(0.5,0)    = 0.5*1 + 0.5*5 = 3
        // (1,1)->(0.5,0.5)  = 0.25*(1+3+5+7) = 4
        // (2,2)->(1,1)      = 7
        assertEquals(1f, yd[0 * outW + 0], 1e-5);
        assertEquals(2f, yd[0 * outW + 1], 1e-5);
        assertEquals(3f, yd[1 * outW + 0], 1e-5);
        assertEquals(4f, yd[1 * outW + 1], 1e-5);
        assertEquals(7f, yd[2 * outW + 2], 1e-5);
    }

    @Test
    public void border_vs_zeros_padding_out_of_range() {
        float[] xd = new float[]{1, 2, 3, 4};
        NDArray x = new ConcreteNDArray(xd, new Shape(1, 2, 2, 1));

        // One sample well outside the image at (-1, -1).
        float[] gd = new float[]{-1f, -1f};
        NDArray g = new ConcreteNDArray(gd, new Shape(1, 1, 1, 2));

        NDArray zeros = GridSampling.gridSample2dForward(x, g, "zeros");
        assertEquals(0f, zeros.toFloatArray()[0], 1e-5);

        NDArray border = GridSampling.gridSample2dForward(x, g, "border");
        // Clamped to (0, 0) -> the pixel value 1.
        assertEquals(1f, border.toFloatArray()[0], 1e-5);
    }

    @Test
    public void numerical_gradient_check_wrt_input() {
        Random rng = new Random(7);
        int B = 1, H = 3, W = 3, C = 1;
        float[] xd = new float[B * H * W * C];
        for (int i = 0; i < xd.length; i++) xd[i] = (rng.nextFloat() - 0.5f);
        Shape xShape = new Shape(B, H, W, C);

        int outH = 3, outW = 3;
        float[] gd = new float[B * outH * outW * 2];
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                int base = (oy * outW + ox) * 2;
                // Non-integer sample positions to exercise the four-corner blend.
                gd[base] = oy + 0.3f;
                gd[base + 1] = ox + 0.7f;
            }
        }
        NDArray grid = new ConcreteNDArray(gd, new Shape(B, outH, outW, 2));

        NDArray x0 = new ConcreteNDArray(xd.clone(), xShape);

        Map<String, PyTree> paramMap = new LinkedHashMap<>();
        paramMap.put("x", PyTree.leaf(x0));
        PyTree params = PyTree.dict(paramMap);

        java.util.function.Function<PyTree, NDArray> loss = pt -> {
            NDArray xx = ((PyTree.Leaf) ((PyTree.MapNode) pt).children().get("x")).value();
            NDArray y = GridSampling.gridSample2dTraced(xx, grid, "border");
            return y.mul(y).sum();
        };

        PyTree grads = Grad.gradTree(loss).apply(params);
        NDArray dX = ((PyTree.Leaf) ((PyTree.MapNode) grads).children().get("x")).value();
        float[] analytic = dX.toFloatArray();

        float eps = 1e-3f;
        for (int i = 0; i < xd.length; i++) {
            float[] xp = xd.clone(); xp[i] += eps;
            float[] xm = xd.clone(); xm[i] -= eps;
            float lp = lossValue(new ConcreteNDArray(xp, xShape), grid);
            float lm = lossValue(new ConcreteNDArray(xm, xShape), grid);
            float num = (lp - lm) / (2f * eps);
            assertEquals(num, analytic[i], 1e-2f, "input[" + i + "] grad mismatch");
        }
    }

    private static float lossValue(NDArray x, NDArray grid) {
        NDArray y = GridSampling.gridSample2dForward(x, grid, "border");
        float[] yd = y.toFloatArray();
        float sum = 0f;
        for (float v : yd) sum += v * v;
        return sum;
    }
}
