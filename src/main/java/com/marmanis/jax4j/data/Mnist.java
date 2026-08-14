package com.marmanis.jax4j.data;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.pytree.PyTree;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

/**
 * Loader for the classic MNIST handwritten digits dataset.
 * Downloads the dataset files from a public mirror, caches them locally under
 * {@code ~/.jax4j/datasets/mnist/}, and parses the idx-ubyte binary format.
 */
public final class Mnist {
    private Mnist() {}

    private static final String MIRROR_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    public static class MnistDataset implements Dataset {
        private final float[] xsData;
        private final float[] ysData;
        private final int size;
        private final int imageSize;

        public MnistDataset(float[] xsData, float[] ysData, int imageSize) {
            this.xsData = xsData;
            this.ysData = ysData;
            this.imageSize = imageSize;
            this.size = ysData.length;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public PyTree get(int index) {
            float[] img = java.util.Arrays.copyOfRange(xsData, index * imageSize, (index + 1) * imageSize);
            float label = ysData[index];
            NDArray x = new ConcreteNDArray(img, new Shape(imageSize));
            NDArray y = new ConcreteNDArray(new float[]{label}, new Shape());
            return PyTree.list(PyTree.leaf(x), PyTree.leaf(y));
        }

        /** Returns the entire dataset inputs concatenated as a single 2D array. */
        public NDArray getXs() {
            return new ConcreteNDArray(xsData, new Shape(size, imageSize));
        }

        /** Returns the entire dataset targets concatenated as a single 1D array. */
        public NDArray getYs() {
            return new ConcreteNDArray(ysData, new Shape(size));
        }
    }

    /**
     * Loads either the training split or testing split of MNIST.
     * Downloads and caches the required binary files if they are not present locally.
     *
     * @param train If true, loads the training split (60,000 items); otherwise loads test split (10,000 items).
     */
    public static MnistDataset load(boolean train) throws IOException, InterruptedException {
        String imagesFile = train ? TRAIN_IMAGES : TEST_IMAGES;
        String labelsFile = train ? TRAIN_LABELS : TEST_LABELS;

        Path cacheDir = Paths.get(System.getProperty("user.home"), ".jax4j", "datasets", "mnist");
        Path imagesPath = cacheDir.resolve(imagesFile);
        Path labelsPath = cacheDir.resolve(labelsFile);

        downloadFile(MIRROR_URL + imagesFile, imagesPath);
        downloadFile(MIRROR_URL + labelsFile, labelsPath);

        byte[] decompressedImages = decompressGzip(imagesPath);
        byte[] decompressedLabels = decompressGzip(labelsPath);

        return parse(decompressedImages, decompressedLabels);
    }

    /**
     * Parses decompressed IDX binary format buffers into a MnistDataset.
     */
    public static MnistDataset parse(byte[] imagesBytes, byte[] labelsBytes) throws IOException {
        ByteBuffer imgBuf = ByteBuffer.wrap(imagesBytes);
        int imgMagic = imgBuf.getInt();
        if (imgMagic != 2051) {
            throw new IOException("Invalid MNIST images magic number: " + imgMagic);
        }
        int numImages = imgBuf.getInt();
        int rows = imgBuf.getInt();
        int cols = imgBuf.getInt();
        int imageSize = rows * cols;

        float[] xsData = new float[numImages * imageSize];
        for (int i = 0; i < xsData.length; i++) {
            xsData[i] = (imgBuf.get() & 0xFF) / 255.0f;
        }

        ByteBuffer lblBuf = ByteBuffer.wrap(labelsBytes);
        int lblMagic = lblBuf.getInt();
        if (lblMagic != 2049) {
            throw new IOException("Invalid MNIST labels magic number: " + lblMagic);
        }
        int numLabels = lblBuf.getInt();
        if (numLabels != numImages) {
            throw new IOException("Mismatch between images count (" + numImages + ") and labels count (" + numLabels + ")");
        }

        float[] ysData = new float[numLabels];
        for (int i = 0; i < numLabels; i++) {
            ysData[i] = lblBuf.get() & 0xFF;
        }

        return new MnistDataset(xsData, ysData, imageSize);
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

    private static byte[] decompressGzip(Path gzipFile) throws IOException {
        try (GZIPInputStream gis = new GZIPInputStream(Files.newInputStream(gzipFile));
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int len;
            while ((len = gis.read(buffer)) > 0) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        }
    }
}
