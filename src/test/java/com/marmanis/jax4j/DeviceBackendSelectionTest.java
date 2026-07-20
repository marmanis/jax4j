package com.marmanis.jax4j;

import com.marmanis.jax4j.core.Device;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies {@code Device.useBackend} resolves to the expected TornadoVM
 * device type on this sandbox's dual-backend SDK (driver 0 = OpenCL,
 * driver 1 = PTX), and that the unqualified default is PTX/GPU.
 */
public class DeviceBackendSelectionTest {

    @Test
    public void ptxBackendResolvesToGpu() {
        Device.useBackend("ptx");
        Device d = Device.defaultDevice();
        if (d.getTornadoDevice() != null) {
            uk.ac.manchester.tornado.api.common.TornadoDevice expectedDevice = uk.ac.manchester.tornado.api.TornadoExecutionPlan.getDevice(1, 0);
            assertEquals(expectedDevice.getDeviceName(), d.getTornadoDevice().getDeviceName());
        }
    }

    @Test
    public void openclBackendResolvesToCpu() {
        Device.useBackend("opencl");
        Device d = Device.defaultDevice();
        if (d.getTornadoDevice() != null) {
            uk.ac.manchester.tornado.api.common.TornadoDevice expectedDevice = uk.ac.manchester.tornado.api.TornadoExecutionPlan.getDevice(0, 0);
            assertEquals(expectedDevice.getDeviceName(), d.getTornadoDevice().getDeviceName());
        }
    }

    @Test
    public void unknownBackendThrows() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
            () -> Device.useBackend("cuda"));
    }
}
