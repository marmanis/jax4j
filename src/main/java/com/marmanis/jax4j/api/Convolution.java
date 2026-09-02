package com.marmanis.jax4j.api;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ir.Conv2DMeta;
import com.marmanis.jax4j.ir.Conv2DTransposeMeta;
import com.marmanis.jax4j.ir.Conv3DMeta;
import com.marmanis.jax4j.ir.DepthwiseConv2DMeta;
import com.marmanis.jax4j.ir.Equation;
import com.marmanis.jax4j.ir.Pool2DMeta;
import com.marmanis.jax4j.ir.Pool3DMeta;
import com.marmanis.jax4j.ir.Primitive;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.List;

/**
 * Channels-last 2-D convolution and pooling: shape math, {@code im2col}/{@code col2im},
 * and eager forward/backward kernels. All ops assume input layout
 * {@code [batch, height, width, channels]} and (for convolution) kernel layout
 * {@code [kH, kW, inC, outC]}.
 *
 * <p>The public forward entry points ({@link #conv2dForward},
 * {@link #maxPool2dForward}, {@link #avgPool2dForward}) transparently emit a
 * traced equation when a {@link Tracer} is active and at least one input is a
 * {@link TracedNDArray}, and fall back to the pure {@code *Eager} kernels
 * otherwise. This lets these ops appear as a single {@link Primitive#CONV2D}
 * (or {@link Primitive#MAX_POOL_2D}/{@link Primitive#AVG_POOL_2D}) equation
 * inside {@code Grad}'s jaxpr, so autodiff dispatches straight to the matching
 * {@code *Backward} routine below.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public final class Convolution {
    private Convolution() {}

    // ---- shape math -------------------------------------------------------

    /** Output size for one spatial axis given input size, kernel size, stride, and padding mode. */
    public static int outputDim(int inDim, int kernelDim, int stride, String padding) {
        return switch (padding) {
            case "same"  -> (inDim + stride - 1) / stride;
            case "valid" -> (inDim - kernelDim + stride) / stride;
            default -> throw new IllegalArgumentException("Unknown padding: " + padding);
        };
    }

    /**
     * Per-axis padding amounts {@code [padTop, padBottom, padLeft, padRight]}
     * for {@code "same"} (asymmetric when the total is odd, matching Keras:
     * top/left get the smaller half) or all zeros for {@code "valid"}.
     */
    public static int[] paddingAmounts(int inH, int inW, int kH, int kW, int strideH, int strideW, String padding) {
        if ("valid".equals(padding)) return new int[]{0, 0, 0, 0};
        if ("same".equals(padding)) {
            int outH = outputDim(inH, kH, strideH, "same");
            int outW = outputDim(inW, kW, strideW, "same");
            int totalH = Math.max(0, (outH - 1) * strideH + kH - inH);
            int totalW = Math.max(0, (outW - 1) * strideW + kW - inW);
            int padTop = totalH / 2;
            int padLeft = totalW / 2;
            return new int[]{ padTop, totalH - padTop, padLeft, totalW - padLeft };
        }
        throw new IllegalArgumentException("Unknown padding: " + padding);
    }

    // ---- conv2d -----------------------------------------------------------

    /** Traced-or-eager forward: {@code y = conv2d(input, kernel, strides, padding)}. */
    public static NDArray conv2dForward(NDArray input, NDArray kernel, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        boolean traced = tracer != null && (input instanceof TracedNDArray || kernel instanceof TracedNDArray);
        if (!traced) return conv2dEager(input, kernel, strides, padding);

        Var inVar = toVar(input, tracer);
        Var kVar = toVar(kernel, tracer);
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int outH = outputDim(inDims[1], kDims[0], strides[0], padding);
        int outW = outputDim(inDims[2], kDims[1], strides[1], padding);
        Shape outShape = new Shape(inDims[0], outH, outW, kDims[3]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        tracer.addEquation(new Equation(
            List.of(inVar, kVar), List.of(outVar),
            Primitive.CONV2D,
            new Conv2DMeta(strides.clone(), padding)));
        return new TracedNDArray(outVar);
    }

    /**
     * Pure eager im2col-based conv2d forward. Used directly and as the CONV2D
     * executor body.
     *
     * <p>The {@code cols @ ker} matmul dispatches through
     * {@link com.marmanis.jax4j.backend.ExecutionBackend#forDevice}, so a
     * kernel or input placed on a TornadoVM device routes the multiply to the
     * GPU (which owns a plan-cached matmul kernel). Host tensors keep the
     * existing sparse-aware loop, which stays faster than a dense CPU matmul
     * on padded feature maps.
     */
    public static NDArray conv2dEager(NDArray input, NDArray kernel, int[] strides, String padding) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        if (inDims.length != 4 || kDims.length != 4) {
            throw new IllegalArgumentException("conv2d expects rank-4 input and kernel");
        }
        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int kH = kDims[0], kW = kDims[1], inC = kDims[2], outC = kDims[3];
        if (inC != C) throw new IllegalArgumentException("kernel inC " + inC + " != input C " + C);
        int outH = outputDim(H, kH, strides[0], padding);
        int outW = outputDim(W, kW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, kH, kW, strides[0], strides[1], padding);

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] cols = im2col(in, B, H, W, C, kH, kW, strides, pad);
        int rows = B * outH * outW;
        int K = kH * kW * C;

        // Route to the device-appropriate backend. When either the input or
        // the kernel lives on a Tornado device, TornadoVMBackend.matmul owns
        // a cached GPU kernel that is many-fold faster than the host loop
        // once the plan warms up; on host, we keep the sparse-aware inner
        // loop below because it beats a dense CPU matmul on padded tensors.
        com.marmanis.jax4j.core.Device device = input.device();
        if (kernel.device().getTornadoDevice() != null) device = kernel.device();

        if (device.getTornadoDevice() != null) {
            float[] out = com.marmanis.jax4j.backend.ExecutionBackend.forDevice(device)
                    .matmul(cols, ker, rows, K, outC, device);
            return new ConcreteNDArray(out, new Shape(B, outH, outW, outC),
                    input.dtype(), device);
        }

        float[] out = new float[rows * outC];
        for (int r = 0; r < rows; r++) {
            int cbase = r * K;
            int obase = r * outC;
            for (int kk = 0; kk < K; kk++) {
                float v = cols[cbase + kk];
                if (v == 0f) continue;
                int kBase = kk * outC;
                for (int oc = 0; oc < outC; oc++) {
                    out[obase + oc] += v * ker[kBase + oc];
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, outC));
    }

    /**
     * Returns {@code [gradInput, gradKernel]} for conv2d.
     *
     * <p>Same device-aware routing as {@link #conv2dEager}: when any of
     * {@code input}, {@code kernel}, or {@code gradOutput} lives on a
     * TornadoVM device, the two internal matmul-shaped reductions
     * ({@code gKer = colsᵀ @ gOut} and {@code gCols = gOut @ kerᵀ}) go to
     * that device's plan-cached GPU matmul. Host tensors keep the existing
     * fused sparse-aware loop, which folds both reductions into a single
     * pass and skips zero-valued gradient elements.
     */
    public static NDArray[] conv2dBackward(NDArray input, NDArray kernel, NDArray gradOutput,
                                           int[] strides, String padding) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int kH = kDims[0], kW = kDims[1], outC = kDims[3];
        int outH = outputDim(H, kH, strides[0], padding);
        int outW = outputDim(W, kW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, kH, kW, strides[0], strides[1], padding);
        int K = kH * kW * C;
        int rows = B * outH * outW;

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] gOut = gradOutput.toFloatArray();
        float[] cols = im2col(in, B, H, W, C, kH, kW, strides, pad);

        // Pick the target device from any GPU-resident operand.
        com.marmanis.jax4j.core.Device device = input.device();
        if (kernel.device().getTornadoDevice() != null) device = kernel.device();
        if (gradOutput.device().getTornadoDevice() != null) device = gradOutput.device();

        if (device.getTornadoDevice() != null) {
            // Two dense matmuls via the device backend, each with a cheap
            // linear transpose on the host beforehand.
            float[] colsT = transpose2d(cols, rows, K);         // [K, rows]
            float[] kerT = transpose2d(ker, K, outC);           // [outC, K]

            com.marmanis.jax4j.backend.ExecutionBackend backend =
                    com.marmanis.jax4j.backend.ExecutionBackend.forDevice(device);
            float[] gKer  = backend.matmul(colsT, gOut, K,    rows, outC, device); // [K, outC]
            float[] gCols = backend.matmul(gOut,  kerT, rows, outC, K,    device); // [rows, K]

            float[] gIn = col2im(gCols, B, H, W, C, kH, kW, strides, pad);
            return new NDArray[]{
                new ConcreteNDArray(gIn,  input.shape(),  input.dtype(), device),
                new ConcreteNDArray(gKer, kernel.shape(), input.dtype(), device)
            };
        }

        // Host path: fused loop that folds both matmuls into one row scan and
        // skips zero-valued gOut entries (typical after ReLU + padding).
        float[] gKer = new float[K * outC];
        float[] gCols = new float[rows * K];
        for (int r = 0; r < rows; r++) {
            int cbase = r * K;
            int obase = r * outC;
            for (int oc = 0; oc < outC; oc++) {
                float g = gOut[obase + oc];
                if (g == 0f) continue;
                for (int kk = 0; kk < K; kk++) {
                    gKer[kk * outC + oc] += cols[cbase + kk] * g;
                }
            }
            for (int kk = 0; kk < K; kk++) {
                float s = 0f;
                int kBase = kk * outC;
                for (int oc = 0; oc < outC; oc++) {
                    s += gOut[obase + oc] * ker[kBase + oc];
                }
                gCols[cbase + kk] = s;
            }
        }
        float[] gIn = col2im(gCols, B, H, W, C, kH, kW, strides, pad);
        return new NDArray[]{
            new ConcreteNDArray(gIn, input.shape()),
            new ConcreteNDArray(gKer, kernel.shape())
        };
    }

    /** Row-major 2-D transpose of {@code a[rows, cols]} into {@code out[cols, rows]}. */
    private static float[] transpose2d(float[] a, int rows, int cols) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int rowBase = r * cols;
            for (int c = 0; c < cols; c++) {
                out[c * rows + r] = a[rowBase + c];
            }
        }
        return out;
    }

    // ---- pool -------------------------------------------------------------

    public static NDArray maxPool2dForward(NDArray input, int[] poolSize, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        if (tracer != null && input instanceof TracedNDArray t) {
            return emitPool(tracer, t, poolSize, strides, padding, true);
        }
        return maxPool2dEager(input, poolSize, strides, padding);
    }

    public static NDArray avgPool2dForward(NDArray input, int[] poolSize, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        if (tracer != null && input instanceof TracedNDArray t) {
            return emitPool(tracer, t, poolSize, strides, padding, false);
        }
        return avgPool2dEager(input, poolSize, strides, padding);
    }

    private static NDArray emitPool(Tracer tracer, TracedNDArray input,
                                    int[] poolSize, int[] strides, String padding, boolean isMax) {
        int[] d = input.shape().dimensions();
        int outH = outputDim(d[1], poolSize[0], strides[0], padding);
        int outW = outputDim(d[2], poolSize[1], strides[1], padding);
        Shape outShape = new Shape(d[0], outH, outW, d[3]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        Primitive p = isMax ? Primitive.MAX_POOL_2D : Primitive.AVG_POOL_2D;
        tracer.addEquation(new Equation(
            List.of(input.getVar()), List.of(outVar), p,
            new Pool2DMeta(poolSize.clone(), strides.clone(), padding, isMax)));
        return new TracedNDArray(outVar);
    }

    public static NDArray maxPool2dEager(NDArray input, int[] poolSize, int[] strides, String padding) {
        int[] d = input.shape().dimensions();
        int B = d[0], H = d[1], W = d[2], C = d[3];
        int pH = poolSize[0], pW = poolSize[1];
        int outH = outputDim(H, pH, strides[0], padding);
        int outW = outputDim(W, pW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, pH, pW, strides[0], strides[1], padding);
        float[] in = input.toFloatArray();
        float[] out = new float[B * outH * outW * C];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    for (int c = 0; c < C; c++) {
                        float best = Float.NEGATIVE_INFINITY;
                        boolean any = false;
                        for (int i = 0; i < pH; i++) {
                            int ih = ihBase + i;
                            if (ih < 0 || ih >= H) continue;
                            for (int j = 0; j < pW; j++) {
                                int iw = iwBase + j;
                                if (iw < 0 || iw >= W) continue;
                                float v = in[((b * H + ih) * W + iw) * C + c];
                                if (v > best) best = v;
                                any = true;
                            }
                        }
                        out[((b * outH + oh) * outW + ow) * C + c] = any ? best : 0f;
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, C));
    }

    public static NDArray maxPool2dBackward(NDArray input, NDArray gradOutput,
                                            int[] poolSize, int[] strides, String padding) {
        int[] d = input.shape().dimensions();
        int B = d[0], H = d[1], W = d[2], C = d[3];
        int pH = poolSize[0], pW = poolSize[1];
        int outH = outputDim(H, pH, strides[0], padding);
        int outW = outputDim(W, pW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, pH, pW, strides[0], strides[1], padding);
        float[] in = input.toFloatArray();
        float[] g = gradOutput.toFloatArray();
        float[] gIn = new float[in.length];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    for (int c = 0; c < C; c++) {
                        float best = Float.NEGATIVE_INFINITY;
                        int argIdx = -1;
                        for (int i = 0; i < pH; i++) {
                            int ih = ihBase + i;
                            if (ih < 0 || ih >= H) continue;
                            for (int j = 0; j < pW; j++) {
                                int iw = iwBase + j;
                                if (iw < 0 || iw >= W) continue;
                                int idx = ((b * H + ih) * W + iw) * C + c;
                                float v = in[idx];
                                if (v > best) { best = v; argIdx = idx; }
                            }
                        }
                        if (argIdx >= 0) {
                            gIn[argIdx] += g[((b * outH + oh) * outW + ow) * C + c];
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(gIn, input.shape());
    }

    public static NDArray avgPool2dEager(NDArray input, int[] poolSize, int[] strides, String padding) {
        int[] d = input.shape().dimensions();
        int B = d[0], H = d[1], W = d[2], C = d[3];
        int pH = poolSize[0], pW = poolSize[1];
        int outH = outputDim(H, pH, strides[0], padding);
        int outW = outputDim(W, pW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, pH, pW, strides[0], strides[1], padding);
        float invN = 1f / (pH * pW);
        float[] in = input.toFloatArray();
        float[] out = new float[B * outH * outW * C];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    for (int c = 0; c < C; c++) {
                        float sum = 0f;
                        for (int i = 0; i < pH; i++) {
                            int ih = ihBase + i;
                            if (ih < 0 || ih >= H) continue;
                            for (int j = 0; j < pW; j++) {
                                int iw = iwBase + j;
                                if (iw < 0 || iw >= W) continue;
                                sum += in[((b * H + ih) * W + iw) * C + c];
                            }
                        }
                        out[((b * outH + oh) * outW + ow) * C + c] = sum * invN;
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, C));
    }

    public static NDArray avgPool2dBackward(NDArray gradOutput, int[] inputShape,
                                            int[] poolSize, int[] strides, String padding) {
        int B = inputShape[0], H = inputShape[1], W = inputShape[2], C = inputShape[3];
        int pH = poolSize[0], pW = poolSize[1];
        int outH = outputDim(H, pH, strides[0], padding);
        int outW = outputDim(W, pW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, pH, pW, strides[0], strides[1], padding);
        float invN = 1f / (pH * pW);
        float[] g = gradOutput.toFloatArray();
        float[] gIn = new float[B * H * W * C];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    for (int c = 0; c < C; c++) {
                        float gv = g[((b * outH + oh) * outW + ow) * C + c] * invN;
                        for (int i = 0; i < pH; i++) {
                            int ih = ihBase + i;
                            if (ih < 0 || ih >= H) continue;
                            for (int j = 0; j < pW; j++) {
                                int iw = iwBase + j;
                                if (iw < 0 || iw >= W) continue;
                                gIn[((b * H + ih) * W + iw) * C + c] += gv;
                            }
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(gIn, new Shape(B, H, W, C));
    }

    // ---- depthwise conv2d -------------------------------------------------

    /** Traced-or-eager forward for depthwise convolution. */
    public static NDArray depthwiseConv2dForward(NDArray input, NDArray kernel,
                                                 int[] strides, String padding, int depthMultiplier) {
        Tracer tracer = Tracer.current();
        boolean traced = tracer != null && (input instanceof TracedNDArray || kernel instanceof TracedNDArray);
        if (!traced) return depthwiseConv2dEager(input, kernel, strides, padding, depthMultiplier);

        Var inVar = toVar(input, tracer);
        Var kVar = toVar(kernel, tracer);
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int outH = outputDim(inDims[1], kDims[0], strides[0], padding);
        int outW = outputDim(inDims[2], kDims[1], strides[1], padding);
        Shape outShape = new Shape(inDims[0], outH, outW, inDims[3] * depthMultiplier);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        tracer.addEquation(new Equation(
            List.of(inVar, kVar), List.of(outVar),
            Primitive.DEPTHWISE_CONV_2D,
            new DepthwiseConv2DMeta(strides.clone(), padding, depthMultiplier)));
        return new TracedNDArray(outVar);
    }

    /** Pure eager depthwise conv2d forward. Each input channel is convolved
     * with its own {@code depthMultiplier} filters; output has
     * {@code inC * depthMultiplier} channels. */
    public static NDArray depthwiseConv2dEager(NDArray input, NDArray kernel,
                                               int[] strides, String padding, int depthMultiplier) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        if (inDims.length != 4 || kDims.length != 4) {
            throw new IllegalArgumentException("depthwise_conv2d expects rank-4 input and kernel");
        }
        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int kH = kDims[0], kW = kDims[1], kInC = kDims[2], kDepth = kDims[3];
        if (kInC != C) throw new IllegalArgumentException("kernel inC " + kInC + " != input C " + C);
        if (kDepth != depthMultiplier) throw new IllegalArgumentException(
            "kernel depth " + kDepth + " != depthMultiplier " + depthMultiplier);
        int outH = outputDim(H, kH, strides[0], padding);
        int outW = outputDim(W, kW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, kH, kW, strides[0], strides[1], padding);
        int outC = C * depthMultiplier;

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] out = new float[B * outH * outW * outC];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    int outBase = ((b * outH + oh) * outW + ow) * outC;
                    for (int i = 0; i < kH; i++) {
                        int ih = ihBase + i;
                        if (ih < 0 || ih >= H) continue;
                        for (int j = 0; j < kW; j++) {
                            int iw = iwBase + j;
                            if (iw < 0 || iw >= W) continue;
                            int inBase = ((b * H + ih) * W + iw) * C;
                            int kbase = ((i * kW + j) * C) * kDepth;
                            for (int c = 0; c < C; c++) {
                                float v = in[inBase + c];
                                if (v == 0f) continue;
                                int kOff = kbase + c * kDepth;
                                int outOff = outBase + c * kDepth;
                                for (int m = 0; m < kDepth; m++) {
                                    out[outOff + m] += v * ker[kOff + m];
                                }
                            }
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, outC));
    }

    /** Returns {@code [gradInput, gradKernel]} for depthwise conv2d. */
    public static NDArray[] depthwiseConv2dBackward(NDArray input, NDArray kernel, NDArray gradOutput,
                                                    int[] strides, String padding, int depthMultiplier) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int B = inDims[0], H = inDims[1], W = inDims[2], C = inDims[3];
        int kH = kDims[0], kW = kDims[1], kDepth = kDims[3];
        int outH = outputDim(H, kH, strides[0], padding);
        int outW = outputDim(W, kW, strides[1], padding);
        int[] pad = paddingAmounts(H, W, kH, kW, strides[0], strides[1], padding);
        int outC = C * kDepth;

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] gOut = gradOutput.toFloatArray();
        float[] gIn = new float[B * H * W * C];
        float[] gKer = new float[kH * kW * C * kDepth];

        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    int outBase = ((b * outH + oh) * outW + ow) * outC;
                    for (int i = 0; i < kH; i++) {
                        int ih = ihBase + i;
                        if (ih < 0 || ih >= H) continue;
                        for (int j = 0; j < kW; j++) {
                            int iw = iwBase + j;
                            if (iw < 0 || iw >= W) continue;
                            int inBase = ((b * H + ih) * W + iw) * C;
                            int kbase = ((i * kW + j) * C) * kDepth;
                            for (int c = 0; c < C; c++) {
                                int kOff = kbase + c * kDepth;
                                int outOff = outBase + c * kDepth;
                                float xv = in[inBase + c];
                                float gAcc = 0f;
                                for (int m = 0; m < kDepth; m++) {
                                    float g = gOut[outOff + m];
                                    gKer[kOff + m] += xv * g;
                                    gAcc += g * ker[kOff + m];
                                }
                                gIn[inBase + c] += gAcc;
                            }
                        }
                    }
                }
            }
        }
        return new NDArray[]{
            new ConcreteNDArray(gIn, input.shape()),
            new ConcreteNDArray(gKer, kernel.shape())
        };
    }

    // ---- conv2d transpose -------------------------------------------------

    /** Output spatial size for one axis of a 2-D transposed convolution. */
    public static int outputDimTranspose(int inDim, int kernelDim, int stride, String padding) {
        return switch (padding) {
            case "valid" -> (inDim - 1) * stride + kernelDim;
            case "same"  -> inDim * stride;
            default -> throw new IllegalArgumentException("Unknown padding: " + padding);
        };
    }

    /** Traced-or-eager forward for transposed convolution. Kernel layout is
     * {@code [kH, kW, filters, inC]} (Keras convention — swapped channels vs
     * regular {@link #conv2dForward}). */
    public static NDArray conv2dTransposeForward(NDArray input, NDArray kernel,
                                                 int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        boolean traced = tracer != null && (input instanceof TracedNDArray || kernel instanceof TracedNDArray);
        if (!traced) return conv2dTransposeEager(input, kernel, strides, padding);

        Var inVar = toVar(input, tracer);
        Var kVar = toVar(kernel, tracer);
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int outH = outputDimTranspose(inDims[1], kDims[0], strides[0], padding);
        int outW = outputDimTranspose(inDims[2], kDims[1], strides[1], padding);
        Shape outShape = new Shape(inDims[0], outH, outW, kDims[2]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        tracer.addEquation(new Equation(
            List.of(inVar, kVar), List.of(outVar),
            Primitive.CONV_2D_TRANSPOSE,
            new Conv2DTransposeMeta(strides.clone(), padding)));
        return new TracedNDArray(outVar);
    }

    /** Pure eager transposed conv2d forward. */
    public static NDArray conv2dTransposeEager(NDArray input, NDArray kernel,
                                               int[] strides, String padding) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        if (inDims.length != 4 || kDims.length != 4) {
            throw new IllegalArgumentException("conv2d_transpose expects rank-4 input and kernel");
        }
        int B = inDims[0], iH = inDims[1], iW = inDims[2], inC = inDims[3];
        int kH = kDims[0], kW = kDims[1], filters = kDims[2], kInC = kDims[3];
        if (kInC != inC) throw new IllegalArgumentException(
            "kernel inC (last axis) " + kInC + " != input C " + inC);
        int outH = outputDimTranspose(iH, kH, strides[0], padding);
        int outW = outputDimTranspose(iW, kW, strides[1], padding);
        int outH_full = (iH - 1) * strides[0] + kH;
        int outW_full = (iW - 1) * strides[1] + kW;
        int padTop = (outH_full - outH) / 2;
        int padLeft = (outW_full - outW) / 2;

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] out = new float[B * outH * outW * filters];

        for (int b = 0; b < B; b++) {
            for (int iy = 0; iy < iH; iy++) {
                for (int ix = 0; ix < iW; ix++) {
                    int inBase = ((b * iH + iy) * iW + ix) * inC;
                    for (int kh = 0; kh < kH; kh++) {
                        int oy = iy * strides[0] + kh - padTop;
                        if (oy < 0 || oy >= outH) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int ox = ix * strides[1] + kw - padLeft;
                            if (ox < 0 || ox >= outW) continue;
                            int outBase = ((b * outH + oy) * outW + ox) * filters;
                            int kbase = ((kh * kW + kw) * filters) * inC;
                            for (int oc = 0; oc < filters; oc++) {
                                float acc = 0f;
                                int kOff = kbase + oc * inC;
                                for (int ic = 0; ic < inC; ic++) {
                                    acc += in[inBase + ic] * ker[kOff + ic];
                                }
                                out[outBase + oc] += acc;
                            }
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outH, outW, filters));
    }

    /** Returns {@code [gradInput, gradKernel]} for conv2d_transpose. */
    public static NDArray[] conv2dTransposeBackward(NDArray input, NDArray kernel, NDArray gradOutput,
                                                    int[] strides, String padding) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int B = inDims[0], iH = inDims[1], iW = inDims[2], inC = inDims[3];
        int kH = kDims[0], kW = kDims[1], filters = kDims[2];
        int outH = outputDimTranspose(iH, kH, strides[0], padding);
        int outW = outputDimTranspose(iW, kW, strides[1], padding);
        int outH_full = (iH - 1) * strides[0] + kH;
        int outW_full = (iW - 1) * strides[1] + kW;
        int padTop = (outH_full - outH) / 2;
        int padLeft = (outW_full - outW) / 2;

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] gOut = gradOutput.toFloatArray();
        float[] gIn = new float[B * iH * iW * inC];
        float[] gKer = new float[kH * kW * filters * inC];

        for (int b = 0; b < B; b++) {
            for (int iy = 0; iy < iH; iy++) {
                for (int ix = 0; ix < iW; ix++) {
                    int inBase = ((b * iH + iy) * iW + ix) * inC;
                    for (int kh = 0; kh < kH; kh++) {
                        int oy = iy * strides[0] + kh - padTop;
                        if (oy < 0 || oy >= outH) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int ox = ix * strides[1] + kw - padLeft;
                            if (ox < 0 || ox >= outW) continue;
                            int outBase = ((b * outH + oy) * outW + ox) * filters;
                            int kbase = ((kh * kW + kw) * filters) * inC;
                            for (int oc = 0; oc < filters; oc++) {
                                float g = gOut[outBase + oc];
                                if (g == 0f) continue;
                                int kOff = kbase + oc * inC;
                                for (int ic = 0; ic < inC; ic++) {
                                    gIn[inBase + ic] += g * ker[kOff + ic];
                                    gKer[kOff + ic] += g * in[inBase + ic];
                                }
                            }
                        }
                    }
                }
            }
        }
        return new NDArray[]{
            new ConcreteNDArray(gIn, input.shape()),
            new ConcreteNDArray(gKer, kernel.shape())
        };
    }

    // ---- im2col / col2im --------------------------------------------------

    /**
     * Flattens {@code input} into a matrix whose rows are one receptive-field
     * patch each. Rows: {@code B * outH * outW}. Columns: {@code kH * kW * C},
     * ordered {@code (i, j, c)} — same layout as the kernel's leading three
     * axes when it is viewed as {@code [K, outC]}.
     */
    public static float[] im2col(float[] input, int B, int H, int W, int C, int kH, int kW, int[] strides, int[] pad) {
        int outH = outputDimFromPad(H, kH, strides[0], pad[0], pad[1]);
        int outW = outputDimFromPad(W, kW, strides[1], pad[2], pad[3]);
        int K = kH * kW * C;
        int rows = B * outH * outW;
        float[] cols = new float[rows * K];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    int rowBase = ((b * outH + oh) * outW + ow) * K;
                    for (int i = 0; i < kH; i++) {
                        int ih = ihBase + i;
                        for (int j = 0; j < kW; j++) {
                            int iw = iwBase + j;
                            int kbase = rowBase + (i * kW + j) * C;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            int inBase = ((b * H + ih) * W + iw) * C;
                            System.arraycopy(input, inBase, cols, kbase, C);
                        }
                    }
                }
            }
        }
        return cols;
    }

    /**
     * Inverse of {@link #im2col} for the backward pass: scatters each patch row
     * back to its input location and <b>accumulates</b> (a single input pixel
     * contributes to multiple output positions, so col2im must sum them).
     */
    public static float[] col2im(float[] cols, int B, int H, int W, int C, int kH, int kW, int[] strides, int[] pad) {
        int outH = outputDimFromPad(H, kH, strides[0], pad[0], pad[1]);
        int outW = outputDimFromPad(W, kW, strides[1], pad[2], pad[3]);
        int K = kH * kW * C;
        float[] in = new float[B * H * W * C];
        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < outH; oh++) {
                int ihBase = oh * strides[0] - pad[0];
                for (int ow = 0; ow < outW; ow++) {
                    int iwBase = ow * strides[1] - pad[2];
                    int rowBase = ((b * outH + oh) * outW + ow) * K;
                    for (int i = 0; i < kH; i++) {
                        int ih = ihBase + i;
                        if (ih < 0 || ih >= H) continue;
                        for (int j = 0; j < kW; j++) {
                            int iw = iwBase + j;
                            if (iw < 0 || iw >= W) continue;
                            int inBase = ((b * H + ih) * W + iw) * C;
                            int kbase = rowBase + (i * kW + j) * C;
                            for (int c = 0; c < C; c++) in[inBase + c] += cols[kbase + c];
                        }
                    }
                }
            }
        }
        return in;
    }

    private static int outputDimFromPad(int inDim, int kernelDim, int stride, int padBefore, int padAfter) {
        return (inDim + padBefore + padAfter - kernelDim) / stride + 1;
    }

    private static Var toVar(NDArray a, Tracer tracer) {
        if (a instanceof TracedNDArray t) return t.getVar();
        return tracer.nextConstant(a);
    }

    // =====================================================================
    // 3-D convolution and pooling — channels-last [B, D, H, W, C].
    //
    // These forwards trace to CONV3D / MAX_POOL_3D / AVG_POOL_3D equations so
    // JIT re-execution and eager prediction both work; the CONV3D backward
    // rule in Grad is not yet implemented, so training a model that contains
    // a 3-D layer will fail cleanly with "no backward for conv3d". Reach for
    // {@code model.predict}/{@code evaluate} for now.
    // =====================================================================

    /** Per-axis {@code [padDBefore, padDAfter, padHBefore, padHAfter, padWBefore, padWAfter]} for 3-D. */
    public static int[] paddingAmounts3d(int inD, int inH, int inW,
                                         int kD, int kH, int kW,
                                         int strideD, int strideH, int strideW,
                                         String padding) {
        if ("valid".equals(padding)) return new int[]{0, 0, 0, 0, 0, 0};
        if ("same".equals(padding)) {
            int outD = outputDim(inD, kD, strideD, "same");
            int outH = outputDim(inH, kH, strideH, "same");
            int outW = outputDim(inW, kW, strideW, "same");
            int totalD = Math.max(0, (outD - 1) * strideD + kD - inD);
            int totalH = Math.max(0, (outH - 1) * strideH + kH - inH);
            int totalW = Math.max(0, (outW - 1) * strideW + kW - inW);
            int padDb = totalD / 2;
            int padHb = totalH / 2;
            int padWb = totalW / 2;
            return new int[]{padDb, totalD - padDb, padHb, totalH - padHb, padWb, totalW - padWb};
        }
        throw new IllegalArgumentException("Unknown padding: " + padding);
    }

    /** Traced-or-eager forward: {@code y = conv3d(input, kernel, strides, padding)}. */
    public static NDArray conv3dForward(NDArray input, NDArray kernel, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        boolean traced = tracer != null && (input instanceof TracedNDArray || kernel instanceof TracedNDArray);
        if (!traced) return conv3dEager(input, kernel, strides, padding);

        Var inVar = toVar(input, tracer);
        Var kVar = toVar(kernel, tracer);
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        int outD = outputDim(inDims[1], kDims[0], strides[0], padding);
        int outH = outputDim(inDims[2], kDims[1], strides[1], padding);
        int outW = outputDim(inDims[3], kDims[2], strides[2], padding);
        Shape outShape = new Shape(inDims[0], outD, outH, outW, kDims[4]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        tracer.addEquation(new Equation(
            List.of(inVar, kVar), List.of(outVar),
            Primitive.CONV3D,
            new Conv3DMeta(strides.clone(), padding)));
        return new TracedNDArray(outVar);
    }

    /**
     * Pure eager direct 3-D convolution. Input {@code [B, D, H, W, inC]},
     * kernel {@code [kD, kH, kW, inC, outC]}, output
     * {@code [B, outD, outH, outW, outC]}. No im2col — the six-deep loop nest
     * is small and clear; for large workloads the correct fix is a GPU
     * kernel, not a bigger scratch buffer.
     */
    public static NDArray conv3dEager(NDArray input, NDArray kernel, int[] strides, String padding) {
        int[] inDims = input.shape().dimensions();
        int[] kDims = kernel.shape().dimensions();
        if (inDims.length != 5 || kDims.length != 5) {
            throw new IllegalArgumentException("conv3d expects rank-5 input and kernel, got "
                    + input.shape() + " and " + kernel.shape());
        }
        int B = inDims[0], D = inDims[1], H = inDims[2], W = inDims[3], C = inDims[4];
        int kD = kDims[0], kH = kDims[1], kW = kDims[2], inC = kDims[3], outC = kDims[4];
        if (inC != C) throw new IllegalArgumentException("kernel inC " + inC + " != input C " + C);
        int outD = outputDim(D, kD, strides[0], padding);
        int outH = outputDim(H, kH, strides[1], padding);
        int outW = outputDim(W, kW, strides[2], padding);
        int[] pad = paddingAmounts3d(D, H, W, kD, kH, kW, strides[0], strides[1], strides[2], padding);
        int padD = pad[0], padH = pad[2], padW = pad[4];

        float[] in = input.toFloatArray();
        float[] ker = kernel.toFloatArray();
        float[] out = new float[B * outD * outH * outW * outC];

        int inStrideC = 1;
        int inStrideW = C;
        int inStrideH = W * C;
        int inStrideD = H * W * C;
        int inStrideB = D * H * W * C;

        int kStrideOC = 1;
        int kStrideIC = outC;
        int kStrideW = inC * outC;
        int kStrideH = kW * inC * outC;
        int kStrideD = kH * kW * inC * outC;

        int oStrideC = 1;
        int oStrideW = outC;
        int oStrideH = outW * outC;
        int oStrideD = outH * outW * outC;
        int oStrideB = outD * outH * outW * outC;

        for (int b = 0; b < B; b++) {
            for (int od = 0; od < outD; od++) {
                int idBase = od * strides[0] - padD;
                for (int oh = 0; oh < outH; oh++) {
                    int ihBase = oh * strides[1] - padH;
                    for (int ow = 0; ow < outW; ow++) {
                        int iwBase = ow * strides[2] - padW;
                        int oBase = b * oStrideB + od * oStrideD + oh * oStrideH + ow * oStrideW;
                        for (int kd = 0; kd < kD; kd++) {
                            int id = idBase + kd;
                            if (id < 0 || id >= D) continue;
                            for (int kh = 0; kh < kH; kh++) {
                                int ih = ihBase + kh;
                                if (ih < 0 || ih >= H) continue;
                                for (int kw_ = 0; kw_ < kW; kw_++) {
                                    int iw = iwBase + kw_;
                                    if (iw < 0 || iw >= W) continue;
                                    int inBase = b * inStrideB + id * inStrideD + ih * inStrideH + iw * inStrideW;
                                    int kBase = kd * kStrideD + kh * kStrideH + kw_ * kStrideW;
                                    for (int ic = 0; ic < inC; ic++) {
                                        float v = in[inBase + ic * inStrideC];
                                        if (v == 0f) continue;
                                        int krBase = kBase + ic * kStrideIC;
                                        for (int oc = 0; oc < outC; oc++) {
                                            out[oBase + oc * oStrideC] += v * ker[krBase + oc * kStrideOC];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outD, outH, outW, outC));
    }

    // ---- pool3d -----------------------------------------------------------

    public static NDArray maxPool3dForward(NDArray input, int[] poolSize, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        if (tracer != null && input instanceof TracedNDArray t) {
            return emitPool3d(tracer, t, poolSize, strides, padding, true);
        }
        return maxPool3dEager(input, poolSize, strides, padding);
    }

    public static NDArray avgPool3dForward(NDArray input, int[] poolSize, int[] strides, String padding) {
        Tracer tracer = Tracer.current();
        if (tracer != null && input instanceof TracedNDArray t) {
            return emitPool3d(tracer, t, poolSize, strides, padding, false);
        }
        return avgPool3dEager(input, poolSize, strides, padding);
    }

    private static NDArray emitPool3d(Tracer tracer, TracedNDArray input,
                                      int[] poolSize, int[] strides, String padding, boolean isMax) {
        int[] d = input.shape().dimensions();
        int outD = outputDim(d[1], poolSize[0], strides[0], padding);
        int outH = outputDim(d[2], poolSize[1], strides[1], padding);
        int outW = outputDim(d[3], poolSize[2], strides[2], padding);
        Shape outShape = new Shape(d[0], outD, outH, outW, d[4]);
        Var outVar = tracer.nextVar(outShape, input.dtype());
        Primitive p = isMax ? Primitive.MAX_POOL_3D : Primitive.AVG_POOL_3D;
        tracer.addEquation(new Equation(
            List.of(input.getVar()), List.of(outVar), p,
            new Pool3DMeta(poolSize.clone(), strides.clone(), padding, isMax)));
        return new TracedNDArray(outVar);
    }

    public static NDArray maxPool3dEager(NDArray input, int[] poolSize, int[] strides, String padding) {
        return pool3dEager(input, poolSize, strides, padding, true);
    }

    public static NDArray avgPool3dEager(NDArray input, int[] poolSize, int[] strides, String padding) {
        return pool3dEager(input, poolSize, strides, padding, false);
    }

    private static NDArray pool3dEager(NDArray input, int[] poolSize, int[] strides, String padding, boolean isMax) {
        int[] d = input.shape().dimensions();
        if (d.length != 5) {
            throw new IllegalArgumentException("pool3d expects rank-5 input [B,D,H,W,C], got " + input.shape());
        }
        int B = d[0], D = d[1], H = d[2], W = d[3], C = d[4];
        int pD = poolSize[0], pH = poolSize[1], pW = poolSize[2];
        int outD = outputDim(D, pD, strides[0], padding);
        int outH = outputDim(H, pH, strides[1], padding);
        int outW = outputDim(W, pW, strides[2], padding);
        int[] pad = paddingAmounts3d(D, H, W, pD, pH, pW, strides[0], strides[1], strides[2], padding);
        int padD = pad[0], padH = pad[2], padW = pad[4];

        float[] in = input.toFloatArray();
        float[] out = new float[B * outD * outH * outW * C];

        int inStrideC = 1;
        int inStrideW = C;
        int inStrideH = W * C;
        int inStrideD = H * W * C;
        int inStrideB = D * H * W * C;
        int oStrideC = 1;
        int oStrideW = C;
        int oStrideH = outW * C;
        int oStrideD = outH * outW * C;
        int oStrideB = outD * outH * outW * C;

        for (int b = 0; b < B; b++) {
            for (int od = 0; od < outD; od++) {
                int idBase = od * strides[0] - padD;
                for (int oh = 0; oh < outH; oh++) {
                    int ihBase = oh * strides[1] - padH;
                    for (int ow = 0; ow < outW; ow++) {
                        int iwBase = ow * strides[2] - padW;
                        int oBase = b * oStrideB + od * oStrideD + oh * oStrideH + ow * oStrideW;
                        for (int c = 0; c < C; c++) {
                            float acc = isMax ? Float.NEGATIVE_INFINITY : 0f;
                            int count = 0;
                            for (int kd = 0; kd < pD; kd++) {
                                int id = idBase + kd;
                                if (id < 0 || id >= D) continue;
                                for (int kh = 0; kh < pH; kh++) {
                                    int ih = ihBase + kh;
                                    if (ih < 0 || ih >= H) continue;
                                    for (int kw_ = 0; kw_ < pW; kw_++) {
                                        int iw = iwBase + kw_;
                                        if (iw < 0 || iw >= W) continue;
                                        float v = in[b * inStrideB + id * inStrideD + ih * inStrideH + iw * inStrideW + c * inStrideC];
                                        if (isMax) { if (v > acc) acc = v; }
                                        else       { acc += v; count++; }
                                    }
                                }
                            }
                            if (!isMax) acc = count == 0 ? 0f : acc / count;
                            else if (acc == Float.NEGATIVE_INFINITY) acc = 0f;
                            out[oBase + c * oStrideC] = acc;
                        }
                    }
                }
            }
        }
        return new ConcreteNDArray(out, new Shape(B, outD, outH, outW, C));
    }
}
