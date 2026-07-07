package com.marmanis.jax4j.ir;

/**
 * Atomic operations in jax4j.
 */
public enum Primitive {
    ADD("add"),
    SUB("sub"),
    MUL("mul"),
    DIV("div"),
    DOT("dot"),
    EXP("exp"),
    LOG("log"),
    SIN("sin"),
    COS("cos"),
    SUM("sum"),
    MEAN("mean"),
    SUM_AXIS("sum_axis"),
    MEAN_AXIS("mean_axis"),
    TANH("tanh"),
    RELU("relu"),
    SIGMOID("sigmoid"),
    COND("cond"),
    WHILE("while"),
    SCAN("scan"),
    GT("gt"),
    GE("ge"),
    LT("lt"),
    LE("le"),
    EQ("eq"),
    NE("ne"),
    MAX("max"),
    MIN("min"),
    ARGMAX("argmax"),
    ARGMIN("argmin"),
    PMAP("pmap"),
    PSUM("psum"),
    ALL_GATHER("all_gather"),
    CAST("cast"),
    GATHER("gather"),
    FFI_CALL("ffi_call"),
    CHECKPOINT("checkpoint");

    private final String name;

    Primitive(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
