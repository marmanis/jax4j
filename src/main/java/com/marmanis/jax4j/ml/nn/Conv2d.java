package com.marmanis.jax4j.ml.nn;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.Module;
import com.marmanis.jax4j.ml.Static;
import com.marmanis.jax4j.api.Vmap;
import com.marmanis.jax4j.core.Device;

public record Conv2d(
    NDArray weight,
    NDArray bias,
    @Static int inC,
    @Static int outC,
    @Static int kH,
    @Static int kW,
    @Static int stride
) implements Module {

    public static Conv2d init(PRNGKey key, int inChannels, int outChannels, int kernelSize, int stride) {
        return init(key, inChannels, outChannels, kernelSize, kernelSize, stride);
    }

    public static Conv2d init(PRNGKey key, int inChannels, int outChannels, int kH, int kW, int stride) {
        int fanIn = inChannels * kH * kW;
        int fanOut = outChannels * kH * kW;
        NDArray w = Random.glorotUniform(key, new Shape(outChannels, inChannels, kH, kW), fanIn, fanOut);
        NDArray b = new ConcreteNDArray(new float[outChannels], new Shape(outChannels));
        return new Conv2d(w, b, inChannels, outChannels, kH, kW, stride);
    }

    public NDArray apply(NDArray x) {
        Shape xShape = x.shape();
        if (xShape.rank() == 4) {
            // Batched input: [batch, inC, H, W] -> map over the leading batch dimension
            return Vmap.vmap(this::apply).apply(x);
        }
        if (xShape.rank() != 3) {
            throw new IllegalArgumentException("Conv2d input must be of rank 3 [inChannels, H, W] or rank 4 [batch, inChannels, H, W]");
        }

        int H = xShape.dimensions()[1];
        int W = xShape.dimensions()[2];

        int outH = (H - kH) / stride + 1;
        int outW = (W - kW) / stride + 1;

        int patchSize = inC * kH * kW;
        int[] flatIdx = new int[outH * outW * patchSize];
        int idx = 0;
        for (int oh = 0; oh < outH; oh++) {
            int startH = oh * stride;
            for (int ow = 0; ow < outW; ow++) {
                int startW = ow * stride;
                for (int c = 0; c < inC; c++) {
                    int cOffset = c * H * W;
                    for (int kh = 0; kh < kH; kh++) {
                        int hOffset = (startH + kh) * W;
                        for (int kw = 0; kw < kW; kw++) {
                            flatIdx[idx++] = cOffset + hOffset + (startW + kw);
                        }
                    }
                }
            }
        }

        NDArray indices = new ConcreteNDArray(flatIdx, new Shape(outH * outW, patchSize), x.device());
        NDArray xFlat = x.reshape(new Shape(inC * H * W, 1));
        NDArray patches = com.marmanis.jax4j.api.Numpy.take(xFlat, indices);
        patches = patches.reshape(new Shape(outH * outW, patchSize));

        NDArray wMat = weight.reshape(new Shape(outC, patchSize)).transpose(1, 0);
        NDArray out = patches.dot(wMat);

        out = out.reshape(new Shape(outH, outW, outC)).transpose(2, 0, 1);

        if (bias != null) {
            out = out.add(bias.reshape(new Shape(outC, 1, 1)));
        }
        return out;
    }
}
