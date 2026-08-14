package com.marmanis.jax4j.data;

import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;

/**
 * Utility class for generating synthetic datasets (classification & regression problems).
 */
public final class Generators {
    private Generators() {}

    public record RegressionResult(NDArray xs, NDArray ys, NDArray coef, NDArray intercept) {}

    public record ClassificationResult(NDArray xs, NDArray ys) {}

    /**
     * Generates a random linear regression dataset: {@code Y = X * W + b + noise * epsilon}.
     *
     * @param key       The random key.
     * @param nSamples  Number of samples.
     * @param nFeatures Number of features.
     * @param nTargets  Number of target outputs.
     * @param noise     Standard deviation of Gaussian noise.
     */
    public static RegressionResult makeRegression(PRNGKey key, int nSamples, int nFeatures, int nTargets, float noise) {
        PRNGKey[] keys = Random.split(key, 4);
        NDArray xs = Random.normal(keys[0], new Shape(nSamples, nFeatures));
        NDArray coef = Random.normal(keys[1], new Shape(nFeatures, nTargets));
        NDArray intercept = Random.normal(keys[2], new Shape(nTargets));

        NDArray ys = xs.dot(coef);
        ys = ys.add(intercept.reshape(new Shape(1, nTargets)));

        if (noise > 0f) {
            NDArray epsilon = Random.normal(keys[3], new Shape(nSamples, nTargets));
            NDArray noiseScalar = new ConcreteNDArray(new float[]{noise}, new Shape());
            ys = ys.add(epsilon.mul(noiseScalar));
        }

        return new RegressionResult(xs, ys, coef, intercept);
    }

    /**
     * Generates a 2D classification dataset with two interleaving half-circles (moons).
     *
     * @param key      The random key.
     * @param nSamples Total number of samples.
     * @param noise    Standard deviation of Gaussian noise.
     */
    public static ClassificationResult makeMoons(PRNGKey key, int nSamples, float noise) {
        float[] x = new float[nSamples];
        float[] y = new float[nSamples];
        float[] labels = new float[nSamples];

        PRNGKey[] keys = Random.split(key, 3);
        float[] theta = Random.uniform(keys[0], new Shape(nSamples), 0.0f, (float) Math.PI).toFloatArray();
        float[] noiseX = noise > 0f ? Random.normal(keys[1], new Shape(nSamples)).toFloatArray() : null;
        float[] noiseY = noise > 0f ? Random.normal(keys[2], new Shape(nSamples)).toFloatArray() : null;

        int half = nSamples / 2;
        for (int i = 0; i < nSamples; i++) {
            float th = theta[i];
            float nx = noiseX != null ? noiseX[i] * noise : 0f;
            float ny = noiseY != null ? noiseY[i] * noise : 0f;
            if (i < half) {
                x[i] = (float) Math.cos(th) + nx;
                y[i] = (float) Math.sin(th) + ny;
                labels[i] = 0.0f;
            } else {
                x[i] = 1.0f - (float) Math.cos(th) + nx;
                y[i] = 0.5f - (float) Math.sin(th) + ny;
                labels[i] = 1.0f;
            }
        }

        float[] features = new float[nSamples * 2];
        for (int i = 0; i < nSamples; i++) {
            features[2 * i] = x[i];
            features[2 * i + 1] = y[i];
        }

        NDArray xs = new ConcreteNDArray(features, new Shape(nSamples, 2));
        NDArray ys = new ConcreteNDArray(labels, new Shape(nSamples));
        return new ClassificationResult(xs, ys);
    }

    /**
     * Generates a 2D classification dataset with concentric outer and inner circles.
     *
     * @param key      The random key.
     * @param nSamples Total number of samples.
     * @param noise    Standard deviation of Gaussian noise.
     * @param factor   Scale factor between inner and outer circles (e.g. 0.8).
     */
    public static ClassificationResult makeCircles(PRNGKey key, int nSamples, float noise, float factor) {
        float[] x = new float[nSamples];
        float[] y = new float[nSamples];
        float[] labels = new float[nSamples];

        PRNGKey[] keys = Random.split(key, 3);
        float[] theta = Random.uniform(keys[0], new Shape(nSamples), 0.0f, (float) (2.0 * Math.PI)).toFloatArray();
        float[] noiseX = noise > 0f ? Random.normal(keys[1], new Shape(nSamples)).toFloatArray() : null;
        float[] noiseY = noise > 0f ? Random.normal(keys[2], new Shape(nSamples)).toFloatArray() : null;

        int half = nSamples / 2;
        for (int i = 0; i < nSamples; i++) {
            float th = theta[i];
            float nx = noiseX != null ? noiseX[i] * noise : 0f;
            float ny = noiseY != null ? noiseY[i] * noise : 0f;
            if (i < half) {
                x[i] = (float) Math.cos(th) + nx;
                y[i] = (float) Math.sin(th) + ny;
                labels[i] = 0.0f;
            } else {
                x[i] = factor * (float) Math.cos(th) + nx;
                y[i] = factor * (float) Math.sin(th) + ny;
                labels[i] = 1.0f;
            }
        }

        float[] features = new float[nSamples * 2];
        for (int i = 0; i < nSamples; i++) {
            features[2 * i] = x[i];
            features[2 * i + 1] = y[i];
        }

        NDArray xs = new ConcreteNDArray(features, new Shape(nSamples, 2));
        NDArray ys = new ConcreteNDArray(labels, new Shape(nSamples));
        return new ClassificationResult(xs, ys);
    }
}
