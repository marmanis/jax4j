package com.marmanis.jax4j.keras;

import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.keras.callbacks.Callback;
import com.marmanis.jax4j.keras.losses.Loss;
import com.marmanis.jax4j.keras.metrics.Metric;
import com.marmanis.jax4j.keras.optimizers.Optimizer;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.pytree.PyTrees;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Functional-API Keras model: takes one or more symbolic {@link KerasTensor}
 * inputs and outputs and walks the resulting DAG in topological order at
 * {@link #call} / {@link #fit} / {@link #evaluate} / {@link #predict} time.
 *
 * <p>Under the hood, training is a pure function of the model's parameter
 * {@link PyTree}: {@link #fit} freezes weights into a flat {@code MapNode}
 * (keys {@code "layerName/paramName"}), builds a {@code Function<PyTree,
 * NDArray>} loss closure per batch, hands it to
 * {@link Grad#gradTree(java.util.function.Function)}, and asks the
 * {@link Optimizer} for the next params. Layers stay mutable at the user
 * level; the {@link #syncParamsToLayers} pass copies the trained values back
 * so eager {@link #predict}/{@link Layer#call(NDArray)} keep working.
 * @author <a href="mailto:babis@marmanis.com">Babis Marmanis</a>
 */
public class Model {

    protected List<KerasTensor> inputs = new ArrayList<>();
    protected List<KerasTensor> outputs = new ArrayList<>();

    /** Layers in topological order (same order as {@link #nodesInTopoOrder}). */
    protected List<Layer> layersInTopoOrder = new ArrayList<>();
    /** DAG nodes in topological order — {@code null} for input placeholders. */
    protected List<KerasTensor.LayerNode> nodesInTopoOrder = new ArrayList<>();

    protected boolean graphBuilt = false;
    protected long buildSeed = 42L;

    protected Optimizer optimizer;
    protected Loss lossFn;
    protected List<Metric> metrics = List.of();

    private boolean stopTraining = false;

    protected Model() {}

    public Model(KerasTensor input, KerasTensor output) {
        this(List.of(input), List.of(output));
    }

    public Model(List<KerasTensor> inputs, List<KerasTensor> outputs) {
        this.inputs = new ArrayList<>(inputs);
        this.outputs = new ArrayList<>(outputs);
        ensureBuilt();
    }

    protected final void ensureBuilt() {
        if (graphBuilt) return;
        if (inputs.isEmpty() || outputs.isEmpty()) return;
        topoSortAndBuild();
        graphBuilt = true;
    }

    /**
     * Rebuild the topological order and re-run each layer's {@link Layer#build}.
     * Called whenever {@link Sequential} mutates its layer list.
     */
    protected final void rebuild() {
        graphBuilt = false;
        layersInTopoOrder.clear();
        nodesInTopoOrder.clear();
        ensureBuilt();
    }

    private void topoSortAndBuild() {
        List<KerasTensor.LayerNode> nodes = new ArrayList<>();
        Set<KerasTensor.LayerNode> seen = new LinkedHashSet<>();
        for (KerasTensor out : outputs) collectNodes(out, seen, nodes);

        PRNGKey base = new PRNGKey(buildSeed);
        PRNGKey[] keys = Random.split(base, Math.max(1, nodes.size()));

        int idx = 0;
        for (KerasTensor.LayerNode node : nodes) {
            Layer layer = node.layer();
            Shape inputShape = node.inputs().get(0).shape();
            layer.build(inputShape, keys[idx++]);
            layersInTopoOrder.add(layer);
            nodesInTopoOrder.add(node);
        }
    }

    private void collectNodes(KerasTensor t, Set<KerasTensor.LayerNode> seen, List<KerasTensor.LayerNode> ordered) {
        if (t.source() instanceof KerasTensor.LayerNode ln) {
            if (seen.contains(ln)) return;
            for (KerasTensor inp : ln.inputs()) collectNodes(inp, seen, ordered);
            seen.add(ln);
            ordered.add(ln);
        }
    }

    /** Flatten this model's params to a PyTree.MapNode keyed by "layerName/paramName". */
    public PyTree collectParams() {
        Map<String, PyTree> children = new LinkedHashMap<>();
        for (Layer layer : layersInTopoOrder) {
            for (String pn : layer.getParamNames()) {
                children.put(layer.getName() + "/" + pn, PyTree.leaf(layer.getParams().get(pn)));
            }
        }
        return PyTree.dict(children);
    }

    /** Convert a params PyTree back to a flat {@code Map} for layer overrides. */
    private static Map<String, NDArray> paramsTreeToMap(PyTree tree) {
        if (!(tree instanceof PyTree.MapNode mn)) throw new IllegalStateException("params tree must be MapNode");
        Map<String, NDArray> out = new LinkedHashMap<>();
        for (Map.Entry<String, PyTree> e : mn.children().entrySet()) {
            if (e.getValue() instanceof PyTree.Leaf leaf) out.put(e.getKey(), leaf.value());
            else throw new IllegalStateException("expected leaf under " + e.getKey());
        }
        return out;
    }

    /** Copy a trained params map back into each layer's mutable {@code params}. */
    protected void syncParamsToLayers(Map<String, NDArray> paramMap) {
        for (Layer layer : layersInTopoOrder) {
            for (String pn : layer.getParamNames()) {
                NDArray v = paramMap.get(layer.getName() + "/" + pn);
                if (v != null) layer.params.put(pn, v);
            }
        }
    }

    /**
     * Compute the model's output for {@code x} using either the layers' own
     * params (when {@code overrides == null}) or the supplied overrides.
     * Subclasses may override to implement a different structure (e.g.
     * {@link Sequential} runs a straight chain).
     */
    protected NDArray forward(NDArray x, Map<String, NDArray> overrides, boolean training) {
        ensureBuilt();
        if (inputs.size() != 1) {
            throw new IllegalStateException("multi-input models require forwardMulti");
        }
        return forwardMulti(List.of(x), overrides, training).get(0);
    }

    protected List<NDArray> forwardMulti(List<NDArray> xs, Map<String, NDArray> overrides, boolean training) {
        ensureBuilt();
        Map<KerasTensor.KerasNode, NDArray> values = new java.util.IdentityHashMap<>();
        for (int i = 0; i < inputs.size(); i++) {
            values.put(inputs.get(i).source(), xs.get(i));
        }
        for (KerasTensor.LayerNode node : nodesInTopoOrder) {
            List<NDArray> nodeInputs = new ArrayList<>(node.inputs().size());
            for (KerasTensor inp : node.inputs()) {
                NDArray v = values.get(inp.source());
                if (v == null) throw new IllegalStateException("missing value for input to " + node.layer().getName());
                nodeInputs.add(v);
            }
            NDArray out = node.layer().call(nodeInputs, overrides, training);
            values.put(node, out);
        }
        List<NDArray> out = new ArrayList<>(outputs.size());
        for (KerasTensor o : outputs) out.add(values.get(o.source()));
        return out;
    }

    /** Eager single-example / single-batch forward pass using this model's own params. */
    public NDArray call(NDArray x) {
        return forward(x, null, false);
    }

    /** Bind training configuration; may be called multiple times. */
    public void compile(Optimizer optimizer, Loss loss, List<Metric> metrics) {
        ensureBuilt();
        this.optimizer = optimizer;
        this.lossFn = loss;
        this.metrics = (metrics == null) ? List.of() : new ArrayList<>(metrics);
    }

    public void compile(Optimizer optimizer, Loss loss) { compile(optimizer, loss, List.of()); }

    public History fit(NDArray x, NDArray y, int epochs, int batchSize) {
        return fit(x, y, epochs, batchSize, List.of());
    }

    public History fit(NDArray x, NDArray y, int epochs, int batchSize, List<Callback> callbacks) {
        if (optimizer == null || lossFn == null) throw new IllegalStateException("call compile() before fit()");
        ensureBuilt();
        stopTraining = false;
        History history = new History();
        callbacks = callbacks == null ? List.of() : callbacks;
        for (Callback cb : callbacks) cb.setModel(this);

        int nExamples = x.shape().dimensions()[0];
        int numBatches = (nExamples + batchSize - 1) / batchSize;

        Map<String, Float> trainBeginLogs = new LinkedHashMap<>();
        trainBeginLogs.put("epochs", (float) epochs);
        for (Callback cb : callbacks) cb.onTrainBegin(trainBeginLogs);

        PyTree params = collectParams();
        Object optState = optimizer.initState(params);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Metric m : metrics) m.reset();
            for (Callback cb : callbacks) cb.onEpochBegin(epoch, new LinkedHashMap<>());

            float epochLossSum = 0f;
            int epochBatchCount = 0;

            for (int b = 0; b < numBatches; b++) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, nExamples);
                NDArray xb = sliceBatch(x, start, end);
                NDArray yb = sliceBatch(y, start, end);

                final PyTree curParams = params;
                final Map<String, NDArray> curMap = paramsTreeToMap(curParams);

                java.util.function.Function<PyTree, NDArray> lossClosure = pt -> {
                    Map<String, NDArray> overrides = paramsTreeToMap(pt);
                    NDArray pred = forwardWithOverrides(xb, overrides, true);
                    return lossFn.call(yb, pred);
                };

                // Eager forward for logging & metrics; separate from the traced pass in lossClosure.
                NDArray predEager = forwardWithOverrides(xb, curMap, false);
                float batchLoss = lossFn.call(yb, predEager).toFloatArray()[0];
                epochLossSum += batchLoss;
                epochBatchCount++;

                for (Metric m : metrics) m.update(yb, predEager);

                PyTree grads = Grad.gradTree(lossClosure).apply(curParams);
                Optimizer.OptimStep step = optimizer.apply(curParams, grads, optState);
                params = step.params();
                optState = step.state();

                Map<String, Float> batchLogs = new LinkedHashMap<>();
                batchLogs.put("loss", batchLoss);
                for (Callback cb : callbacks) cb.onBatchEnd(b, batchLogs);
            }

            syncParamsToLayers(paramsTreeToMap(params));

            Map<String, Float> epochLogs = new LinkedHashMap<>();
            epochLogs.put("loss", epochBatchCount == 0 ? 0f : epochLossSum / epochBatchCount);
            for (Metric m : metrics) epochLogs.put(m.name(), m.result());
            history.record("loss", epochLogs.get("loss"));
            for (Metric m : metrics) history.record(m.name(), m.result());

            for (Callback cb : callbacks) cb.onEpochEnd(epoch, epochLogs);
            if (stopTraining) break;
        }

        for (Callback cb : callbacks) cb.onTrainEnd(new LinkedHashMap<>());
        return history;
    }

    private NDArray forwardWithOverrides(NDArray xb, Map<String, NDArray> overrides, boolean training) {
        return forward(xb, overrides, training);
    }

    public Map<String, Float> evaluate(NDArray x, NDArray y, int batchSize) {
        ensureBuilt();
        for (Metric m : metrics) m.reset();
        int nExamples = x.shape().dimensions()[0];
        int numBatches = (nExamples + batchSize - 1) / batchSize;
        double lossSum = 0.0;
        int batchCount = 0;
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, nExamples);
            NDArray xb = sliceBatch(x, start, end);
            NDArray yb = sliceBatch(y, start, end);
            NDArray pred = forward(xb, null, false);
            if (lossFn != null) {
                lossSum += lossFn.call(yb, pred).toFloatArray()[0];
                batchCount++;
            }
            for (Metric m : metrics) m.update(yb, pred);
        }
        Map<String, Float> out = new LinkedHashMap<>();
        if (lossFn != null) out.put("loss", batchCount == 0 ? 0f : (float) (lossSum / batchCount));
        for (Metric m : metrics) out.put(m.name(), m.result());
        return out;
    }

    public NDArray predict(NDArray x, int batchSize) {
        ensureBuilt();
        int nExamples = x.shape().dimensions()[0];
        int numBatches = (nExamples + batchSize - 1) / batchSize;
        if (numBatches == 1) return forward(x, null, false);
        List<NDArray> chunks = new ArrayList<>(numBatches);
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, nExamples);
            chunks.add(forward(sliceBatch(x, start, end), null, false));
        }
        return concatLeading(chunks);
    }

    public NDArray predict(NDArray x) { return predict(x, x.shape().dimensions()[0]); }

    public int countParams() {
        ensureBuilt();
        int total = 0;
        for (Layer l : layersInTopoOrder) total += l.countParams();
        return total;
    }

    public void summary() { summary(System.out); }

    public void summary(java.io.PrintStream out) {
        ensureBuilt();
        out.println("Model");
        out.println("_________________________________________________________________");
        out.println(String.format("%-25s %-20s %10s", "Layer (name)", "Output Shape", "Param #"));
        out.println("=================================================================");
        for (int i = 0; i < layersInTopoOrder.size(); i++) {
            Layer l = layersInTopoOrder.get(i);
            KerasTensor.LayerNode node = nodesInTopoOrder.get(i);
            Shape outShape = l.computeOutputShape(node.inputs().get(0).shape());
            String type = l.getClass().getSimpleName();
            out.println(String.format("%-25s %-20s %10d",
                type + " (" + l.getName() + ")", outShape.toString(), l.countParams()));
        }
        out.println("=================================================================");
        out.println("Total params: " + countParams());
    }

    /** Called by {@link com.marmanis.jax4j.keras.callbacks.EarlyStopping} to interrupt {@link #fit}. */
    public void stopTraining(boolean stop) { this.stopTraining = stop; }

    public List<Layer> layers() { return Collections.unmodifiableList(layersInTopoOrder); }

    // ----- weight I/O (self-contained little-endian binary format) ---------

    private static final byte[] MAGIC = {'K', 'R', 'S', '1'};

    public void saveWeights(Path path) throws IOException {
        ensureBuilt();
        try (OutputStream out = Files.newOutputStream(path)) {
            out.write(MAGIC);
            writeInt(out, layersInTopoOrder.size());
            for (Layer layer : layersInTopoOrder) {
                writeString(out, layer.getName());
                writeInt(out, layer.getParamNames().size());
                for (String pn : layer.getParamNames()) {
                    writeString(out, pn);
                    writeLeaf(out, layer.getParams().get(pn));
                }
            }
        }
    }

    public void loadWeights(Path path) throws IOException {
        ensureBuilt();
        try (InputStream in = Files.newInputStream(path)) {
            byte[] header = in.readNBytes(4);
            if (header.length != 4 || header[0] != 'K' || header[1] != 'R' || header[2] != 'S' || header[3] != '1') {
                throw new IOException("Bad magic in " + path);
            }
            int nLayers = readInt(in);
            Map<String, NDArray> loaded = new LinkedHashMap<>();
            for (int i = 0; i < nLayers; i++) {
                String layerName = readString(in);
                int nParams = readInt(in);
                for (int j = 0; j < nParams; j++) {
                    String pn = readString(in);
                    loaded.put(layerName + "/" + pn, readLeaf(in));
                }
            }
            syncParamsToLayers(loaded);
        }
    }

    private static void writeInt(OutputStream out, int v) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v);
        out.write(b.array());
    }

    private static int readInt(InputStream in) throws IOException {
        byte[] b = in.readNBytes(4);
        if (b.length != 4) throw new IOException("Truncated");
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    private static void writeString(OutputStream out, String s) throws IOException {
        byte[] data = s.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        writeInt(out, data.length);
        out.write(data);
    }

    private static String readString(InputStream in) throws IOException {
        int n = readInt(in);
        byte[] data = in.readNBytes(n);
        if (data.length != n) throw new IOException("Truncated string");
        return new String(data, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void writeLeaf(OutputStream out, NDArray leaf) throws IOException {
        int[] dims = leaf.shape().dimensions();
        writeInt(out, dtypeCode(leaf.dtype()));
        writeInt(out, dims.length);
        for (int d : dims) writeInt(out, d);
        int size = (int) leaf.shape().size();
        ByteBuffer bb = ByteBuffer.allocate(size * leaf.dtype().byteSize()).order(ByteOrder.LITTLE_ENDIAN);
        switch (leaf.dtype()) {
            case FLOAT32 -> { for (float v : leaf.toFloatArray()) bb.putFloat(v); }
            case FLOAT64 -> { for (double v : leaf.toDoubleArray()) bb.putDouble(v); }
            case INT32   -> { for (int v : leaf.toIntArray()) bb.putInt(v); }
            case INT64   -> { for (long v : leaf.toLongArray()) bb.putLong(v); }
            case BOOL    -> { for (boolean v : leaf.toBoolArray()) bb.put((byte) (v ? 1 : 0)); }
        }
        out.write(bb.array());
    }

    private static NDArray readLeaf(InputStream in) throws IOException {
        int dtypeCode = readInt(in);
        DType dtype = dtypeFromCode(dtypeCode);
        int rank = readInt(in);
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) dims[i] = readInt(in);
        Shape shape = new Shape(dims);
        int size = (int) shape.size();
        int nBytes = size * dtype.byteSize();
        byte[] payload = in.readNBytes(nBytes);
        if (payload.length != nBytes) throw new IOException("Truncated leaf payload");
        ByteBuffer bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        return switch (dtype) {
            case FLOAT32 -> {
                float[] d = new float[size]; for (int i = 0; i < size; i++) d[i] = bb.getFloat();
                yield new ConcreteNDArray(d, shape);
            }
            case FLOAT64 -> {
                double[] d = new double[size]; for (int i = 0; i < size; i++) d[i] = bb.getDouble();
                yield new ConcreteNDArray(d, shape);
            }
            case INT32 -> {
                int[] d = new int[size]; for (int i = 0; i < size; i++) d[i] = bb.getInt();
                yield new ConcreteNDArray(d, shape);
            }
            case INT64 -> {
                long[] d = new long[size]; for (int i = 0; i < size; i++) d[i] = bb.getLong();
                yield new ConcreteNDArray(d, shape);
            }
            case BOOL -> {
                boolean[] d = new boolean[size]; for (int i = 0; i < size; i++) d[i] = bb.get() != 0;
                yield new ConcreteNDArray(d, shape);
            }
        };
    }

    private static int dtypeCode(DType d) {
        return switch (d) { case FLOAT32 -> 1; case FLOAT64 -> 2; case INT32 -> 3; case INT64 -> 4; case BOOL -> 5; };
    }

    private static DType dtypeFromCode(int c) throws IOException {
        return switch (c) {
            case 1 -> DType.FLOAT32; case 2 -> DType.FLOAT64;
            case 3 -> DType.INT32;   case 4 -> DType.INT64; case 5 -> DType.BOOL;
            default -> throw new IOException("Unknown dtype code: " + c);
        };
    }

    // ----- batch slicing helpers --------------------------------------------

    private static NDArray sliceBatch(NDArray x, int start, int end) {
        int[] dims = x.shape().dimensions();
        if (start == 0 && end == dims[0]) return x;
        int rowSize = 1;
        for (int i = 1; i < dims.length; i++) rowSize *= dims[i];
        int rows = end - start;
        int[] outDims = dims.clone();
        outDims[0] = rows;
        Shape outShape = new Shape(outDims);
        return switch (x.dtype()) {
            case FLOAT32 -> {
                float[] src = x.toFloatArray();
                float[] out = new float[rows * rowSize];
                System.arraycopy(src, start * rowSize, out, 0, rows * rowSize);
                yield new ConcreteNDArray(out, outShape);
            }
            case FLOAT64 -> {
                double[] src = x.toDoubleArray();
                double[] out = new double[rows * rowSize];
                System.arraycopy(src, start * rowSize, out, 0, rows * rowSize);
                yield new ConcreteNDArray(out, outShape);
            }
            case INT32 -> {
                int[] src = x.toIntArray();
                int[] out = new int[rows * rowSize];
                System.arraycopy(src, start * rowSize, out, 0, rows * rowSize);
                yield new ConcreteNDArray(out, outShape);
            }
            case INT64 -> {
                long[] src = x.toLongArray();
                long[] out = new long[rows * rowSize];
                System.arraycopy(src, start * rowSize, out, 0, rows * rowSize);
                yield new ConcreteNDArray(out, outShape);
            }
            default -> throw new UnsupportedOperationException("sliceBatch not supported for " + x.dtype());
        };
    }

    private static NDArray concatLeading(List<NDArray> chunks) {
        int rows = 0;
        int[] tail = null;
        for (NDArray c : chunks) {
            int[] d = c.shape().dimensions();
            rows += d[0];
            if (tail == null) tail = d;
        }
        int rowSize = 1;
        for (int i = 1; i < tail.length; i++) rowSize *= tail[i];
        int[] outDims = tail.clone();
        outDims[0] = rows;
        Shape outShape = new Shape(outDims);
        float[] out = new float[rows * rowSize];
        int offset = 0;
        for (NDArray c : chunks) {
            float[] src = c.toFloatArray();
            System.arraycopy(src, 0, out, offset, src.length);
            offset += src.length;
        }
        return new ConcreteNDArray(out, outShape);
    }
}
