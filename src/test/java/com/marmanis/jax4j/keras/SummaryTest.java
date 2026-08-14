package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.layers.Dense;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class SummaryTest {

    @Test
    public void summaryContainsLayerNamesAndParamCount() {
        Layer.resetNameCounters();
        Sequential model = Sequential.of(new Shape(8))
            .add(new Dense(4))
            .add(new Dense(2));
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        model.summary(new PrintStream(buf));
        String out = buf.toString();
        assertTrue(out.contains("dense"), out);
        assertTrue(out.contains("Total params"), out);
        assertTrue(out.contains(String.valueOf(model.countParams())), out);
    }
}
