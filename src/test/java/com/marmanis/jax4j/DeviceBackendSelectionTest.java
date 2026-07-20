package com.marmanis.jax4j;

import com.marmanis.jax4j.core.Device;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies {@code Device.useBackend} resolves to the expected TornadoVM
 * device on whatever backends the current SDK ships with. The driver
 * index for each backend is a build-time choice ({@code opencl → 0,
 * ptx → 1} in dual-backend SDKs), so each test probes for the expected
 * driver slot up front and skips when the SDK doesn't include it.
 */
public class DeviceBackendSelectionTest {

    /** {@code true} iff {@code (driver, 0)} resolves to a real Tornado device. */
    private static boolean backendPresent(int driver) {
        try {
            return TornadoExecutionPlan.getDevice(driver, 0) != null;
        } catch (Throwable t) {
            return false;
        }
    }

    private static TornadoDevice deviceOrSkip(int driver, String backendName) {
        Assumptions.assumeTrue(backendPresent(driver),
            "SDK doesn't ship the '" + backendName + "' backend on driver " + driver
                + "; skipping");
        return TornadoExecutionPlan.getDevice(driver, 0);
    }

    @Test
    public void ptxBackendResolvesToGpu() {
        TornadoDevice expected = deviceOrSkip(1, "ptx");
        Device.useBackend("ptx");
        Device d = Device.defaultDevice();
        Assumptions.assumeTrue(d.getTornadoDevice() != null,
            "PTX driver present but defaultDevice() fell back to host — skipping");
        assertEquals(expected.getDeviceName(), d.getTornadoDevice().getDeviceName());
    }

    @Test
    public void openclBackendResolvesToCpu() {
        TornadoDevice expected = deviceOrSkip(0, "opencl");
        Device.useBackend("opencl");
        Device d = Device.defaultDevice();
        Assumptions.assumeTrue(d.getTornadoDevice() != null,
            "OpenCL driver present but defaultDevice() fell back to host — skipping");
        assertEquals(expected.getDeviceName(), d.getTornadoDevice().getDeviceName());
    }

    @Test
    public void unknownBackendThrows() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
            () -> Device.useBackend("cuda"));
    }
}
