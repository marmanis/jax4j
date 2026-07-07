package com.marmanis.jax4j;

import com.marmanis.jax4j.core.Device;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Device}'s identity semantics and discovery/default fallback.
 */
public class DeviceTest {

    @Test
    public void testHostIsStableSingleton() {
        assertSame(Device.host(), Device.host());
    }

    @Test
    public void testDistinctDeviceInstancesAreNotEqual() {
        Device a = new Device(null);
        Device b = new Device(null);
        assertNotEquals(a, b);
        assertNotSame(a, b);
    }

    @Test
    public void testDeviceEqualsItself() {
        Device a = new Device(null);
        assertEquals(a, a);
    }

    @Test
    public void testDefaultDeviceFallsBackToHostWithoutAccelerators() {
        // Without any discovered accelerator, defaultDevice() must resolve to the
        // host singleton. When a real TornadoVM GPU is discovered (e.g. this
        // sandbox's RTX 4060 under the PTX backend), it must prefer that GPU
        // instead and never silently fall back to the host.
        if (Device.getDevices().isEmpty()) {
            assertSame(Device.host(), Device.defaultDevice());
        } else {
            assertNotSame(Device.host(), Device.defaultDevice());
        }
    }

    @Test
    public void testGetDevicesIsCachedAndStable() {
        List<Device> first = Device.getDevices();
        List<Device> second = Device.getDevices();
        assertSame(first, second);
    }

    @Test
    public void testHostDeviceHasNullTornadoDevice() {
        assertNull(Device.host().getTornadoDevice());
        assertEquals("Host-CPU", Device.host().getName());
    }
}
