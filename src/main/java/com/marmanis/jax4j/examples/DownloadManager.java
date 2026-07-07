package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Port of JAX's {@code examples/datasets.py} to jax4j: downloads the raw
 * MNIST IDX files from the same CVDF mirror, caches them locally, and
 * parses them into {@link NDArray}s (pixels scaled to {@code [0, 1]},
 * labels one-hot encoded), mirroring {@code datasets.mnist_raw()} and
 * {@code datasets.mnist()}.
 */
public class DownloadManager {

    private static final String BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    /** Default cache directory, used when no {@code dataDirectory} is given. */
    public static final Path DATA_DIR = Path.of(System.getProperty("java.io.tmpdir"), "jax_example_data");
    private static final int NUM_CLASSES = 10;

    private record RawImages(byte[] pixels, int numImages, int rows, int cols) {}

    public record MnistData(NDArray trainImages, NDArray trainLabels,
                             NDArray testImages, NDArray testLabels) {}

    /** Downloads {@code url} into {@link #DATA_DIR} as {@code filename}, unless already present. */
    public static void download(String url, String filename) throws IOException {
        download(url, filename, DATA_DIR);
    }

    /** Downloads {@code url} into {@code dataDirectory} as {@code filename}, unless already present. */
    public static void download(String url, String filename, Path dataDirectory) throws IOException {
        if (!Files.exists(dataDirectory)) {
            Files.createDirectories(dataDirectory);
        }
        Path outFile = dataDirectory.resolve(filename);
        if (Files.isRegularFile(outFile)) {
            return;
        }
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder(URI.create(url)).GET().build();
        try {
            HttpResponse<Path> response = client.send(request, HttpResponse.BodyHandlers.ofFile(outFile));
            if (response.statusCode() / 100 != 2) {
                Files.deleteIfExists(outFile);
                throw new IOException("Failed to download " + url + ": HTTP " + response.statusCode());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while downloading " + url, e);
        }
        System.out.println("downloaded " + url + " to " + dataDirectory);
    }

    private static byte[] parseLabels(Path filename) throws IOException {
        try (InputStream in = new GZIPInputStream(new BufferedInputStream(Files.newInputStream(filename)));
             DataInputStream data = new DataInputStream(in)) {
            data.readInt(); // magic
            data.readInt(); // num items
            return data.readAllBytes();
        }
    }

    private static RawImages parseImages(Path filename) throws IOException {
        try (InputStream in = new GZIPInputStream(new BufferedInputStream(Files.newInputStream(filename)));
             DataInputStream data = new DataInputStream(in)) {
            data.readInt(); // magic
            int numData = data.readInt();
            int rows = data.readInt();
            int cols = data.readInt();
            byte[] pixels = data.readAllBytes();
            return new RawImages(pixels, numData, rows, cols);
        }
    }

    /** Downloads and parses the raw MNIST dataset into {@link #DATA_DIR}, mirroring {@code datasets.mnist_raw()}. */
    public static MnistDataRaw mnistRaw() throws IOException {
        return mnistRaw(DATA_DIR);
    }

    /** Downloads and parses the raw MNIST dataset into {@code dataDirectory}, mirroring {@code datasets.mnist_raw()}. */
    public static MnistDataRaw mnistRaw(Path dataDirectory) throws IOException {
        String[] filenames = {
            "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        };
        for (String filename : filenames) {
            download(BASE_URL + filename, filename, dataDirectory);
        }

        RawImages trainImages = parseImages(dataDirectory.resolve("train-images-idx3-ubyte.gz"));
        byte[] trainLabels = parseLabels(dataDirectory.resolve("train-labels-idx1-ubyte.gz"));
        RawImages testImages = parseImages(dataDirectory.resolve("t10k-images-idx3-ubyte.gz"));
        byte[] testLabels = parseLabels(dataDirectory.resolve("t10k-labels-idx1-ubyte.gz"));

        return new MnistDataRaw(trainImages, trainLabels, testImages, testLabels);
    }

    private record MnistDataRaw(RawImages trainImages, byte[] trainLabels,
                                 RawImages testImages, byte[] testLabels) {}

    /** Downloads, parses and normalizes MNIST from {@link #DATA_DIR}, mirroring {@code datasets.mnist(permute_train=False)}. */
    public static MnistData mnist() throws IOException {
        return mnist(false, DATA_DIR);
    }

    /** Downloads, parses and normalizes MNIST from {@link #DATA_DIR}, mirroring {@code datasets.mnist(permute_train)}. */
    public static MnistData mnist(boolean permuteTrain) throws IOException {
        return mnist(permuteTrain, DATA_DIR);
    }

    /** Downloads, parses and normalizes MNIST from {@code dataDirectory}, mirroring {@code datasets.mnist(permute_train)}. */
    public static MnistData mnist(boolean permuteTrain, Path dataDirectory) throws IOException {
        MnistDataRaw raw = mnistRaw(dataDirectory);

        NDArray trainImages = toUnitScaleImages(raw.trainImages());
        NDArray testImages = toUnitScaleImages(raw.testImages());
        NDArray trainLabels = toOneHot(raw.trainLabels());
        NDArray testLabels = toOneHot(raw.testLabels());

        if (permuteTrain) {
            int[] perm = permutation(raw.trainImages().numImages(), new Random(0));
            int imageWidth = raw.trainImages().rows() * raw.trainImages().cols();
            trainImages = permuteRows(trainImages, perm, imageWidth);
            trainLabels = permuteRows(trainLabels, perm, NUM_CLASSES);
        }

        return new MnistData(trainImages, trainLabels, testImages, testLabels);
    }

    /** Flattens each image to a row vector and scales pixels from {@code [0, 255]} to {@code [0, 1]}. */
    private static NDArray toUnitScaleImages(RawImages images) {
        int imageWidth = images.rows() * images.cols();
        float[] data = new float[images.numImages() * imageWidth];
        for (int i = 0; i < data.length; i++) {
            data[i] = (images.pixels()[i] & 0xFF) / 255f;
        }
        return new ConcreteNDArray(data, new Shape(images.numImages(), imageWidth));
    }

    private static NDArray toOneHot(byte[] labels) {
        float[] data = new float[labels.length * NUM_CLASSES];
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i] & 0xFF;
            data[i * NUM_CLASSES + label] = 1f;
        }
        return new ConcreteNDArray(data, new Shape(labels.length, NUM_CLASSES));
    }

    private static int[] permutation(int n, Random rng) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    private static NDArray permuteRows(NDArray array, int[] perm, int rowWidth) {
        float[] src = array.toFloatArray();
        float[] dst = new float[src.length];
        for (int i = 0; i < perm.length; i++) {
            System.arraycopy(src, perm[i] * rowWidth, dst, i * rowWidth, rowWidth);
        }
        return new ConcreteNDArray(dst, array.shape());
    }

    /** Downloads MNIST and prints the resulting shapes, as a smoke test. */
    public static void main(String[] args) throws IOException {
        MnistData data = mnist(true);
        System.out.println("train images: " + data.trainImages().shape());
        System.out.println("train labels: " + data.trainLabels().shape());
        System.out.println("test images: " + data.testImages().shape());
        System.out.println("test labels: " + data.testLabels().shape());
    }
}
