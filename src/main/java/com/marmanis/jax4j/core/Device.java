package com.marmanis.jax4j.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for TornadoVM hardware devices.
 */
public class Device {
    private static final Logger log = LoggerFactory.getLogger(Device.class);

    private static List<Device> cachedDevices = null;
    private static Device defaultDevice = null;
    private static Device hostDevice = null;
    private static boolean discoveryAttempted = false;

    private final TornadoDevice tornadoDevice;

    public Device(TornadoDevice tornadoDevice) {
        this.tornadoDevice = tornadoDevice;
    }

    /**
     * The canonical CPU-fallback device singleton — distinct from
     * {@link #defaultDevice()}, which prefers a GPU when one is discovered.
     * {@code host()} always means "run on the JVM host," regardless of what
     * accelerators are available.
     */
    public static synchronized Device host() {
        if (hostDevice == null) {
            hostDevice = new Device(null);
        }
        return hostDevice;
    }

    public String getName() {
        return tornadoDevice == null ? "Host-CPU" : tornadoDevice.getDeviceName();
    }

    public TornadoDevice getTornadoDevice() {
        return tornadoDevice;
    }

    /**
     * Lists all available TornadoVM devices.
     */
    public static synchronized List<Device> getDevices() {
        if (discoveryAttempted) return cachedDevices;
        
        cachedDevices = new ArrayList<>();
        try {
            // Check first few drivers and devices
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    try {
                        TornadoDevice d = TornadoExecutionPlan.getDevice(i, j);
                        if (d != null) {
                            cachedDevices.add(new Device(d));
                        }
                    } catch (Throwable t) {
                        log.debug("No more TornadoVM devices on driver {} after slot {}: {}", i, j, t.getMessage());
                        break;
                    }
                }
            }
        } catch (Throwable t) {
            log.warn("TornadoVM runtime not available or not configured; falling back to host CPU", t);
        }
        discoveryAttempted = true;
        return cachedDevices;
    }

    /**
     * Returns the default device.
     */
    public static synchronized Device defaultDevice() {
        if (defaultDevice != null) return defaultDevice;

        List<Device> devices = getDevices();
        for (Device d : devices) {
            if (d.getTornadoDevice() != null && d.getTornadoDevice().getDeviceType() == TornadoDeviceType.GPU) {
                defaultDevice = d;
                return d;
            }
        }

        if (!devices.isEmpty()) {
            defaultDevice = devices.get(0);
        } else {
            defaultDevice = host();
        }
        return defaultDevice;
    }

    /**
     * Selects a specific TornadoVM backend ({@code "opencl"} or {@code "ptx"})
     * as the {@link #defaultDevice()} for the remainder of the run, overriding
     * the automatic GPU-preferring selection. TornadoVM's driver indices
     * correspond 1:1 to the backends an SDK was built with, in build order
     * (e.g. {@code --backend opencl,ptx} gives driver 0 = OpenCL, driver 1 =
     * PTX); slot 0 on that driver is used. Falls back to {@link #host()} with
     * a warning if the requested backend has no discoverable device (e.g. a
     * single-backend SDK, or no matching accelerator present).
     */
    public static synchronized void useBackend(String backend) {
        int driver = switch (backend.toLowerCase()) {
            case "opencl" -> 0;
            case "ptx" -> 1;
            default -> throw new IllegalArgumentException(
                "Unknown backend '" + backend + "' (expected 'opencl' or 'ptx')");
        };
        try {
            TornadoDevice d = TornadoExecutionPlan.getDevice(driver, 0);
            if (d != null) {
                defaultDevice = new Device(d);
                return;
            }
        } catch (Throwable t) {
            log.warn("TornadoVM '{}' backend not available; falling back to host CPU", backend, t);
        }
        defaultDevice = host();
    }

    /**
     * Identity-based by design: each {@code Device} instance — whether a real
     * accelerator or a host-CPU placeholder — represents a distinct logical
     * execution target. Two separately constructed {@code Device(null)}
     * instances are intentionally <em>not</em> equal, so tests (and future
     * multi-device code) can stand up multiple synthetic "host-like" devices
     * without real hardware and still tell them apart.
     */
    @Override
    public boolean equals(Object o) {
        return this == o;
    }

    @Override
    public int hashCode() {
        return System.identityHashCode(this);
    }

    @Override
    public String toString() {
        // Include an identity suffix: two distinct Device instances can otherwise
        // print identically (e.g. two synthetic Device(null) test doubles, or two
        // real accelerators of the same model), which would make device-mismatch
        // error messages indistinguishable.
        return "Device[" + getName() + "#" + Integer.toHexString(System.identityHashCode(this)) + "]";
    }
}
