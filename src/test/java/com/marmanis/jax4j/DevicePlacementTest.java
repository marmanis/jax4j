package com.marmanis.jax4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for explicit device placement (`NDArray.to(Device)`) and the
 * device-checked dispatch in {@code ConcreteNDArray}'s elementwise ops.
 *
 * <p>Since this sandbox has no real accelerator, "non-host" devices here are
 * synthetic ({@code new Device(null)}) — distinct logical placements that
 * still wrap a null {@code TornadoDevice}, so combining them exercises the
 * placement/dispatch *logic* (matching devices, backend selection, fallback)
 * without needing real TornadoVM hardware. Real hardware would only change
 * which {@link com.marmanis.jax4j.backend.ExecutionBackend} gets selected.
 */
public class DevicePlacementTest {

    private static NDArray vec(float... v) {
        return new ConcreteNDArray(v, new Shape(v.length));
    }

    @Test
    public void testToReturnsArrayOnRequestedDevice() {
        Device gpu = new Device(null);
        NDArray x = vec(1, 2, 3).to(gpu);
        assertSame(gpu, x.device());
        assertArrayEquals(new float[]{1, 2, 3}, x.toFloatArray());
    }

    @Test
    public void testHostPlusHostStillWorksUnchanged() {
        // Explicitly placed on host() (rather than relying on the default — which
        // prefers a discovered GPU, e.g. this sandbox's RTX 4060 under PTX) so this
        // test exercises the host path regardless of what hardware is available.
        NDArray a = vec(1, 2, 3).to(Device.host());
        NDArray b = vec(4, 5, 6).to(Device.host());
        assertSame(Device.host(), a.device());
        NDArray result = a.add(b);
        assertArrayEquals(new float[]{5, 7, 9}, result.toFloatArray());
        assertSame(Device.host(), result.device());
    }

    @Test
    public void testMismatchedExplicitDevicesThrow() {
        Device gpu0 = new Device(null);
        Device gpu1 = new Device(null);
        NDArray a = vec(1, 2, 3).to(gpu0);
        NDArray b = vec(4, 5, 6).to(gpu1);

        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> a.add(b));
        assertTrue(ex.getMessage().contains("different devices"));
    }

    @Test
    public void testSameExplicitDeviceComputesCorrectly() {
        Device gpu = new Device(null);
        NDArray a = vec(1, 2, 3).to(gpu);
        NDArray b = vec(4, 5, 6).to(gpu);

        NDArray result = a.mul(b);

        assertArrayEquals(new float[]{4, 10, 18}, result.toFloatArray(), 1e-6f);
        assertSame(gpu, result.device());
    }

    @Test
    public void testUnaryOpOnExplicitDeviceComputesCorrectly() {
        Device gpu = new Device(null);
        NDArray x = vec(0f, 1f, 2f).to(gpu);

        NDArray result = x.exp();

        assertEquals((float) Math.exp(0), result.toFloatArray()[0], 1e-5f);
        assertEquals((float) Math.exp(1), result.toFloatArray()[1], 1e-5f);
        assertEquals((float) Math.exp(2), result.toFloatArray()[2], 1e-5f);
        assertSame(gpu, result.device());
    }

    @Test
    public void testBroadcastingOnExplicitDeviceStillComputesAndKeepsDeviceTag() {
        // The TornadoVM backend doesn't support broadcasting; a shape mismatch
        // on a non-host device must still compute correctly (host fallback)
        // while keeping the device tag on the result.
        Device gpu = new Device(null);
        NDArray row = new ConcreteNDArray(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3)).to(gpu);
        NDArray bias = vec(10, 20, 30).to(gpu);

        NDArray result = row.add(bias);

        assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, result.toFloatArray(), 1e-6f);
        assertSame(gpu, result.device());
    }

    @Test
    public void testExplicitlyPlacedHostDeviceIsTreatedAsHost() {
        // Explicitly placing both operands onto Device.host() (rather than relying
        // on the default — which prefers a discovered GPU) must still take the
        // zero-overhead host path and combine fine.
        NDArray a = vec(1, 2, 3).to(Device.host());
        NDArray b = vec(1, 1, 1).to(Device.host());

        NDArray result = a.add(b);

        assertArrayEquals(new float[]{2, 3, 4}, result.toFloatArray(), 1e-6f);
    }
}
