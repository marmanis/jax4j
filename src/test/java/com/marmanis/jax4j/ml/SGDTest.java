package com.marmanis.jax4j.ml;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.ml.nn.Linear;
import com.marmanis.jax4j.ml.optim.Optimizer;
import com.marmanis.jax4j.ml.optim.SGD;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SGDTest {

    @Test
    public void oneStepMovesParamsInNegativeGradientDirection() {
        NDArray w = new ConcreteNDArray(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
        NDArray b = new ConcreteNDArray(new float[]{0f, 0f}, new Shape(2));
        Linear layer = new Linear(w, b);
        Linear grads = new Linear(
            new ConcreteNDArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new Shape(2, 2)),
            new ConcreteNDArray(new float[]{0f, 0f}, new Shape(2)));

        SGD sgd = new SGD(0.5f);
        Optimizer.Update<Linear, SGD.Unit> upd = sgd.update(layer, grads, sgd.init(layer));

        float[] out = upd.newModule().weight().toFloatArray();
        float[] gw = grads.weight().toFloatArray();
        float[] ww = w.toFloatArray();
        for (int i = 0; i < out.length; i++) {
            assertEquals(ww[i] - 0.5f * gw[i], out[i], 1e-6);
        }
    }
}
