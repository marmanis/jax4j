package com.marmanis.jax4j.ir;

/**
 * Atomic operations in jax4j.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
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
    CHECKPOINT("checkpoint"),
    RESHAPE("reshape"),
    TRANSPOSE("transpose"),
    CONCAT("concat"),
    PAD("pad"),
    SCATTER_ADD("scatter_add"),
    FFT("fft"),
    IFFT("ifft"),
    LINALG_SOLVE("linalg_solve"),
    LINALG_SVD("linalg_svd"),
    LINALG_EIG("linalg_eig"),
    CUSTOM_VJP("custom_vjp"),
    CONV2D("conv2d"),
    DEPTHWISE_CONV_2D("depthwise_conv_2d"),
    CONV_2D_TRANSPOSE("conv_2d_transpose"),
    MAX_POOL_2D("max_pool_2d"),
    AVG_POOL_2D("avg_pool_2d"),
    CONV3D("conv3d"),
    MAX_POOL_3D("max_pool_3d"),
    AVG_POOL_3D("avg_pool_3d"),
    SQRT("sqrt"),
    RSQRT("rsqrt"),
    MAX_AXIS("max_axis"),
    MIN_AXIS("min_axis"),
    SLICE("slice"),
    MATMUL("matmul"),
    GRID_SAMPLE_2D("grid_sample_2d"),
    RANDOM_SEED("random_seed"),
    RANDOM_SPLIT("random_split"),
    RANDOM_UNIFORM("random_uniform"),
    RANDOM_NORMAL("random_normal"),
    RANDOM_BERNOULLI("random_bernoulli"),
    RANDOM_PERMUTATION("random_permutation");

    private final String name;

    Primitive(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }
}
