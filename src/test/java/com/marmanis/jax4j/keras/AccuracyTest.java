package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.metrics.Accuracy;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AccuracyTest {

    @Test
    public void accumulatesAcrossBatchesAndResets() {
        Accuracy m = new Accuracy();
        NDArray p1 = new ConcreteNDArray(new float[]{0.9f, 0.1f, 0.2f, 0.8f}, new Shape(2, 2));
        NDArray t1 = new ConcreteNDArray(new int[]{0, 1}, new Shape(2));
        m.update(t1, p1);
        assertEquals(1f, m.result(), 1e-6);

        NDArray p2 = new ConcreteNDArray(new float[]{0.1f, 0.9f, 0.9f, 0.1f}, new Shape(2, 2));
        NDArray t2 = new ConcreteNDArray(new int[]{0, 1}, new Shape(2));
        m.update(t2, p2); // 0 correct out of 2
        assertEquals(0.5f, m.result(), 1e-6);

        m.reset();
        assertEquals(0f, m.result(), 1e-6);
    }

    @Test
    public void oneHotTargetsWorkToo() {
        Accuracy m = new Accuracy();
        NDArray pred = new ConcreteNDArray(new float[]{0.1f, 0.9f, 0.6f, 0.4f}, new Shape(2, 2));
        // pred argmax=[1,0]; targ argmax=[1,0]  -> both correct.
        NDArray targ = new ConcreteNDArray(new float[]{0f, 1f, 1f, 0f}, new Shape(2, 2));
        m.update(targ, pred);
        assertEquals(1.0f, m.result(), 1e-6);
        // Now a mismatch batch: pred argmax=[0,1]; targ argmax=[1,0]  -> 0 correct.
        m.reset();
        NDArray pred2 = new ConcreteNDArray(new float[]{0.9f, 0.1f, 0.4f, 0.6f}, new Shape(2, 2));
        m.update(targ, pred2);
        assertEquals(0.0f, m.result(), 1e-6);
    }
}
