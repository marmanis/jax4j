package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.GridSample2DMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.List;

/**
 * Bilinear grid sampling primitive. Given a channels-last input image
 * {@code [B, H, W, C]} and a per-batch sample grid {@code [B, outH, outW, 2]}
 * whose entries are {@code (y, x)} pixel coordinates (not normalized), returns
 * {@code [B, outH, outW, C]} where each output pixel is the bilinear
 * interpolation of the input at the requested coordinates.
 *
 * <p>Two out-of-range strategies are supported via {@code paddingMode}:
 * {@code "zeros"} treats out-of-range corners as zero; {@code "border"}
 * clamps every sampled coordinate to the nearest valid pixel index.
 *
 * <p>Gradient: only the input image is treated as differentiable. The sample
 * grid always receives a zero cotangent — v1 targets non-learnable geometric
 * transforms (RandomRotation, RandomZoom, UpSampling2D) whose grids are
 * either constant or a stateless random draw, so the grid-VJP is deferred to
 * a later phase.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class GridSampling {
    private GridSampling() {}

    /** Traced-or-eager forward. Emits a {@link Primitive#GRID_SAMPLE_2D}
     *  equation when a tracer is active and either input is a {@link TracedNDArray}. */
    public static NDArray gridSample2dTraced(NDArray input, NDArray grid, String paddingMode) {
        Tracer tracer = Tracer.current();
        boolean traced = tracer != null && (input instanceof TracedNDArray || grid instanceof TracedNDArray);
        if (!traced) return gridSample2dForward(input, grid, paddingMode);

        Var inVar = toVar(input, tracer);
        Var gVar = toVar(grid, tracer);
        int[] inDims = input.shape().dimensions();
        int[] gDims = grid.shape().dimensions();
        Shape outShape = new Shape(inDims[0], gDims[1], gDims[2], inDims[3]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        tracer.addEquation(new Equation(
            List.of(inVar, gVar), List.of(outVar),
            Primitive.GRID_SAMPLE_2D,
            new GridSample2DMeta(paddingMode)));
        return new TracedNDArray(outVar);
    }

    /** Pure eager bilinear sample. */
    public static NDArray gridSample2dForward(NDArray input, NDArray grid, String paddingMode) {
        int[] inDims = input.shape().dimensions();
        int[] gDims = grid.shape().dimensions();
        if (inDims.length != 4) throw new IllegalArgumentException(
            "grid_sample_2d expects rank-4 input [B,H,W,C], got: " + input.shape());
        if (gDims.length != 4 || gDims[3] != 2) throw new IllegalArgumentException(
            "grid_sample_2d expects rank-4 grid [B,outH,outW,2], got: " + grid.shape());
        if (gDims[0] != inDims[0]) throw new IllegalArgumentException(
            "grid batch " + gDims[0] + " != input batch " + inDims[0]);

        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int outH = gDims[1], outW = gDims[2];
        boolean border = "border".equals(paddingMode);
        if (!border && !"zeros".equals(paddingMode)) {
            throw new IllegalArgumentException("paddingMode must be 'zeros' or 'border', got: " + paddingMode);
        }

        float[] in = input.toFloatArray();
        float[] g = grid.toFloatArray();
        float[] out = new float[B * outH * outW * C];

        for (int b = 0; b < B; b++) {
            for (int oy = 0; oy < outH; oy++) {
                for (int ox = 0; ox < outW; ox++) {
                    int gBase = ((b * outH + oy) * outW + ox) * 2;
                    float y = g[gBase];
                    float x = g[gBase + 1];
                    int y0 = (int) Math.floor(y);
                    int x0 = (int) Math.floor(x);
                    int y1 = y0 + 1;
                    int x1 = x0 + 1;
                    float wy = y - y0;
                    float wx = x - x0;
                    float w00 = (1f - wy) * (1f - wx);
                    float w01 = (1f - wy) * wx;
                    float w10 = wy * (1f - wx);
                    float w11 = wy * wx;
                    int outBase = ((b * outH + oy) * outW + ox) * C;
                    accumulate(in, out, b, y0, x0, w00, H, W, C, border, outBase);
                    accumulate(in, out, b, y0, x1, w01, H, W, C, border, outBase);
                    accumulate(in, out, b, y1, x0, w10, H, W, C, border, outBase);
                    accumulate(in, out, b, y1, x1, w11, H, W, C, border, outBase);
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, C));
    }

    private static void accumulate(float[] in, float[] out, int b, int iy, int ix,
                                   float w, int H, int W, int C, boolean border, int outBase) {
        if (w == 0f) return;
        int cy, cx;
        if (border) {
            cy = Math.max(0, Math.min(H - 1, iy));
            cx = Math.max(0, Math.min(W - 1, ix));
        } else {
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) return;
            cy = iy;
            cx = ix;
        }
        int inBase = ((b * H + cy) * W + cx) * C;
        for (int c = 0; c < C; c++) out[outBase + c] += w * in[inBase + c];
    }

    /** VJP w.r.t. the input image. Scatter-adds the bilinear-weighted
     *  {@code gradOutput} into the four surrounding input pixels. */
    public static NDArray gridSample2dBackwardInput(NDArray input, NDArray grid,
                                                    NDArray gradOutput, String paddingMode) {
        int[] inDims = input.shape().dimensions();
        int[] gDims = grid.shape().dimensions();
        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int outH = gDims[1], outW = gDims[2];
        boolean border = "border".equals(paddingMode);

        float[] g = grid.toFloatArray();
        float[] go = gradOutput.toFloatArray();
        float[] gIn = new float[B * H * W * C];

        for (int b = 0; b < B; b++) {
            for (int oy = 0; oy < outH; oy++) {
                for (int ox = 0; ox < outW; ox++) {
                    int gBase = ((b * outH + oy) * outW + ox) * 2;
                    float y = g[gBase];
                    float x = g[gBase + 1];
                    int y0 = (int) Math.floor(y);
                    int x0 = (int) Math.floor(x);
                    int y1 = y0 + 1;
                    int x1 = x0 + 1;
                    float wy = y - y0;
                    float wx = x - x0;
                    float w00 = (1f - wy) * (1f - wx);
                    float w01 = (1f - wy) * wx;
                    float w10 = wy * (1f - wx);
                    float w11 = wy * wx;
                    int outBase = ((b * outH + oy) * outW + ox) * C;
                    scatter(gIn, go, b, y0, x0, w00, H, W, C, border, outBase);
                    scatter(gIn, go, b, y0, x1, w01, H, W, C, border, outBase);
                    scatter(gIn, go, b, y1, x0, w10, H, W, C, border, outBase);
                    scatter(gIn, go, b, y1, x1, w11, H, W, C, border, outBase);
                }
            }
        }
        return new ConcreteNDArray(gIn, input.shape());
    }

    private static void scatter(float[] gIn, float[] go, int b, int iy, int ix,
                                float w, int H, int W, int C, boolean border, int outBase) {
        if (w == 0f) return;
        int cy, cx;
        if (border) {
            cy = Math.max(0, Math.min(H - 1, iy));
            cx = Math.max(0, Math.min(W - 1, ix));
        } else {
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) return;
            cy = iy;
            cx = ix;
        }
        int inBase = ((b * H + cy) * W + cx) * C;
        for (int c = 0; c < C; c++) gIn[inBase + c] += w * go[outBase + c];
    }

    private static Var toVar(NDArray a, Tracer tracer) {
        if (a instanceof TracedNDArray t) return t.getVar();
        return tracer.nextConstant(a);
    }
}
