package com.marmanis.jax4j.examples;

import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Port of JAX linear regression example.
 */
public class LinearRegression {

    public static void main(String[] args) {
        ExampleBackend.selectFromArgs(args);

        // 1. Generate synthetic data
        // y = 2*x + 1
        float[] xData = {1, 2, 3, 4, 5};
        float[] yData = {3, 5, 7, 9, 11};
        NDArray x = new ConcreteNDArray(xData, new Shape(5));
        NDArray y = new ConcreteNDArray(yData, new Shape(5));

        // 2. Define Loss Function
        // loss = mean((x * w + b - y)^2)
        // Since our grad only supports Function<NDArray, NDArray> for now,
        // we optimize w while keeping b fixed, or vice versa.
        // Or we can use a custom wrapper.
        
        final float fixedB = 1.0f;
        Function<NDArray, NDArray> lossFn = w -> {
            // System.out.println("DEBUG: lossFn w class: " + w.getClass().getName());
            // System.out.println("DEBUG: lossFn x class: " + x.getClass().getName());
            NDArray prediction = x.mul(w).add(scalar(fixedB, 5));
            NDArray diff = prediction.sub(y);
            return diff.mul(diff).mean();
        };

        // 3. Optimization Loop
        NDArray w = new ConcreteNDArray(new float[]{0.0f}, new Shape(1));
        float learningRate = 0.01f;

        Function<NDArray, NDArray> gradFn = JAX.grad(lossFn);

        System.out.println("Starting optimization (Target weight: 2.0)...");
        for (int i = 0; i < 50; i++) {
            NDArray loss = lossFn.apply(w);
            NDArray grad = gradFn.apply(w);
            w = w.sub(grad.mul(scalar(learningRate, 1)));
            if (i % 10 == 0) {
                System.out.printf("Step %d: Loss = %.4f, Weight = %.4f\n", 
                    i, loss.toFloatArray()[0], w.toFloatArray()[0]);
            }
        }
        
        System.out.printf("Final weight: %.4f (Expected: 2.0)\n", w.toFloatArray()[0]);
    }

    private static NDArray scalar(float v, int size) {
        float[] d = new float[size];
        Arrays.fill(d, v);
        return new ConcreteNDArray(d, new Shape(size));
    }
}
