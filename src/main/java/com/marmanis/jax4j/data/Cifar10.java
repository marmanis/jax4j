package com.marmanis.jax4j.data;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

/**
 * Loader for the CIFAR-10 image classification dataset.
 * Downloads the binary tarball from Toronto University, caches it locally under
 * {@code ~/.jax4j/datasets/}, and parses the binary batch files on-the-fly
 * using a memory-efficient native tar/gzip stream parser.
 */
public final class Cifar10 {
    private Cifar10() {}

    private static final String DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    private static final String CACHE_FILE = "cifar-10-binary.tar.gz";

    public static class Cifar10Dataset implements Dataset {
        private final float[] xsData;
        private final float[] ysData;
        private final int size;

        public Cifar10Dataset(float[] xsData, float[] ysData) {
            this.xsData = xsData;
            this.ysData = ysData;
            this.size = ysData.length;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public PyTree get(int index) {
            float[] img = java.util.Arrays.copyOfRange(xsData, index * 3072, (index + 1) * 3072);
            float label = ysData[index];
            NDArray x = new ConcreteNDArray(img, new Shape(3, 32, 32));
            NDArray y = new ConcreteNDArray(new float[]{label}, new Shape());
            return PyTree.list(PyTree.leaf(x), PyTree.leaf(y));
        }

        /** Returns the entire dataset inputs concatenated as a single 4D array [size, 3, 32, 32]. */
        public NDArray getXs() {
            return new ConcreteNDArray(xsData, new Shape(size, 3, 32, 32));
        }

        /** Returns the entire dataset targets concatenated as a single 1D array [size]. */
        public NDArray getYs() {
            return new ConcreteNDArray(ysData, new Shape(size));
        }
    }

    /**
     * Loads either the training split or testing split of CIFAR-10.
     * Downloads and caches the required binary tarball if not present locally.
     *
     * @param train If true, loads the training split (50,000 items); otherwise loads test split (10,000 items).
     */
    public static Cifar10Dataset load(boolean train) throws IOException, InterruptedException {
        Path cacheDir = Paths.get(System.getProperty("user.home"), ".jax4j", "datasets");
        Path cachePath = cacheDir.resolve(CACHE_FILE);

        downloadFile(DATASET_URL, cachePath);

        int totalImages = train ? 50000 : 10000;
        float[] xsData = new float[totalImages * 3072];
        float[] ysData = new float[totalImages];

        try (InputStream fileIn = Files.newInputStream(cachePath);
             GZIPInputStream gis = new GZIPInputStream(fileIn)) {
            byte[] header = new byte[512];
            int loadedImages = 0;

            while (readFully(gis, header, 512)) {
                // Check for EOF (empty block header has first byte 0)
                if (header[0] == 0) {
                    break;
                }
                String name = new String(header, 0, 100).trim();
                String sizeStr = new String(header, 124, 12).trim();
                if (sizeStr.isEmpty()) {
                    continue;
                }
                long size = Long.parseLong(sizeStr, 8);
                long paddedSize = ((size + 511) / 512) * 512;

                boolean wantFile = false;
                if (train) {
                    wantFile = name.endsWith("data_batch_1.bin") ||
                               name.endsWith("data_batch_2.bin") ||
                               name.endsWith("data_batch_3.bin") ||
                               name.endsWith("data_batch_4.bin") ||
                               name.endsWith("data_batch_5.bin");
                } else {
                    wantFile = name.endsWith("test_batch.bin");
                }

                if (wantFile) {
                    byte[] fileBytes = readBytes(gis, size);
                    parseBatch(fileBytes, xsData, ysData, loadedImages);
                    loadedImages += 10000;
                    // skip padding
                    skipFully(gis, paddedSize - size);
                } else {
                    skipFully(gis, paddedSize);
                }
            }
        }

        return new Cifar10Dataset(xsData, ysData);
    }

    private static void parseBatch(byte[] bytes, float[] xs, float[] ys, int offset) {
        int recordSize = 3073; // 1 label byte + 3072 image bytes
        for (int i = 0; i < 10000; i++) {
            int byteOffset = i * recordSize;
            ys[offset + i] = bytes[byteOffset] & 0xFF;

            for (int p = 0; p < 3072; p++) {
                xs[(offset + i) * 3072 + p] = (bytes[byteOffset + 1 + p] & 0xFF) / 255.0f;
            }
        }
    }

    private static void downloadFile(String fileUrl, Path dest) throws IOException, InterruptedException {
        if (Files.exists(dest)) {
            return;
        }
        Files.createDirectories(dest.getParent());
        HttpClient client = HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.ALWAYS)
            .build();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(fileUrl))
            .build();
        HttpResponse<Path> response = client.send(request, HttpResponse.BodyHandlers.ofFile(dest));
        if (response.statusCode() != 200) {
            Files.deleteIfExists(dest);
            throw new IOException("Failed to download file from " + fileUrl + ", status code: " + response.statusCode());
        }
    }

    private static boolean readFully(InputStream in, byte[] buf, int length) throws IOException {
        int offset = 0;
        while (offset < length) {
            int read = in.read(buf, offset, length - offset);
            if (read < 0) {
                if (offset == 0) {
                    return false;
                }
                throw new EOFException("Unexpected end of stream while reading tar header");
            }
            offset += read;
        }
        return true;
    }

    private static byte[] readBytes(InputStream in, long size) throws IOException {
        byte[] data = new byte[(int) size];
        int offset = 0;
        while (offset < size) {
            int read = in.read(data, offset, (int) (size - offset));
            if (read < 0) {
                throw new EOFException("Unexpected end of stream while reading tar entry content");
            }
            offset += read;
        }
        return data;
    }

    private static void skipFully(InputStream in, long n) throws IOException {
        long remaining = n;
        while (remaining > 0) {
            long skipped = in.skip(remaining);
            if (skipped <= 0) {
                int b = in.read();
                if (b < 0) {
                    break;
                }
                remaining--;
            } else {
                remaining -= skipped;
            }
        }
    }
}
