namespace HartsyInference.ModelAssets.Onnx;

/// <summary>A parsed ONNX <c>TensorProto</c> initializer: its name, shape, element type, and the absolute byte range of its <c>raw_data</c> blob in the source buffer (so it can be copied straight into native memory with no intermediate allocation).</summary>
public sealed class OnnxTensor
{
    public required string Name { get; init; }
    public required long[] Dims { get; init; }
    /// <summary>ONNX <c>TensorProto.DataType</c> (1 = FLOAT, 10 = FLOAT16, 7 = INT64, 6 = INT32, …).</summary>
    public required int DataType { get; init; }
    public required int RawOffset { get; init; }
    public required int RawLength { get; init; }
}

/// <summary>A parsed ONNX <c>NodeProto</c> in graph (topological) order: its op type, inputs, outputs, and optional name. Used to bind anonymized initializers (<c>onnx::MatMul_NNNN</c>) to logical layers by walking which node consumes them.</summary>
public sealed class OnnxNode
{
    public required string OpType { get; init; }
    public required string Name { get; init; }
    public required string[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
}

/// <summary>A graph nested inside a node's attribute — the <c>then_branch</c>/<c>else_branch</c> of an <c>If</c>, or a loop <c>body</c>.
/// <para>Some exports keep their weights here rather than in the top-level graph. Silero VAD is the case that forced this: it wraps the whole network in an <c>If</c> on sample rate, and its fifteen tensors are anonymous <c>Constant</c> nodes inside each branch, invisible to a reader that only walks <c>graph.initializer</c>.</para></summary>
public sealed class OnnxSubgraph
{
    /// <summary>The attribute holding it: <c>then_branch</c>, <c>else_branch</c>, <c>body</c>.</summary>
    public required string AttributeName { get; init; }

    /// <summary>Name of the node that owns it, for diagnostics.</summary>
    public required string NodeName { get; init; }

    /// <summary>Initializers declared inside this branch.</summary>
    public required IReadOnlyList<OnnxTensor> Initializers { get; init; }

    /// <summary>Values of the branch's <c>Constant</c> nodes, in graph order. These are unnamed in the wire format — a <c>Constant</c> carries a tensor, not a weight name — so order is all a caller has to bind them by.</summary>
    public required IReadOnlyList<OnnxTensor> Constants { get; init; }

    public required IReadOnlyList<OnnxNode> Nodes { get; init; }
}

/// <summary>The subset of an ONNX model the engine needs: the initializer tensors (weights) and the node list in graph order. Parsed by walking the protobuf wire format directly (see <see cref="ProtoReader"/>); only <c>ModelProto.graph</c> → {<c>node</c>, <c>initializer</c>} are decoded, every other field is skipped.</summary>
public sealed class OnnxModel
{
    // ModelProto field numbers.
    private const int ModelGraph = 7;
    // GraphProto field numbers.
    private const int GraphNode = 1;
    private const int GraphInitializer = 5;
    // NodeProto field numbers.
    private const int NodeInput = 1;
    private const int NodeOutput = 2;
    private const int NodeName = 3;
    private const int NodeOpType = 4;
    private const int NodeAttribute = 5;
    // AttributeProto field numbers.
    private const int AttrName = 1;
    private const int AttrTensor = 5;
    private const int AttrGraph = 6;
    // TensorProto field numbers.
    private const int TensorDims = 1;
    private const int TensorDataType = 2;
    private const int TensorName = 8;
    private const int TensorRawData = 9;

    public IReadOnlyList<OnnxTensor> Initializers { get; }
    public IReadOnlyList<OnnxNode> Nodes { get; }

    /// <summary>Graphs nested in node attributes, in the order encountered. Empty for the ordinary exports where every weight is a top-level initializer.</summary>
    public IReadOnlyList<OnnxSubgraph> Subgraphs { get; }

    private OnnxModel(List<OnnxTensor> initializers, List<OnnxNode> nodes, List<OnnxSubgraph> subgraphs)
    {
        Initializers = initializers;
        Nodes = nodes;
        Subgraphs = subgraphs;
    }

    /// <summary>Parses the <c>ModelProto</c> in <paramref name="buffer"/> (the full <c>.onnx</c> file bytes).</summary>
    public static OnnxModel Parse(ReadOnlySpan<byte> buffer)
    {
        List<OnnxTensor> inits = [];
        List<OnnxNode> nodes = [];
        List<OnnxSubgraph> subgraphs = [];
        ProtoReader r = new(buffer);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (field == ModelGraph && wire == ProtoReader.WireLengthDelimited)
            {
                (int off, int len) = r.ReadLengthDelimited();
                ParseGraph(buffer, off, off + len, inits, nodes, subgraphs, null);
            }
            else
            {
                r.Skip(wire);
            }
        }
        return new OnnxModel(inits, nodes, subgraphs);
    }

    /// <summary>Parses one GraphProto. <paramref name="constants"/> is non-null only for a nested graph, where the values of <c>Constant</c> nodes are collected because that is where such an export keeps its weights.</summary>
    private static void ParseGraph(ReadOnlySpan<byte> buffer, int start, int end,
        List<OnnxTensor> inits, List<OnnxNode> nodes, List<OnnxSubgraph> subgraphs, List<OnnxTensor>? constants)
    {
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (wire != ProtoReader.WireLengthDelimited) { r.Skip(wire); continue; }
            (int off, int len) = r.ReadLengthDelimited();
            if (field == GraphNode) nodes.Add(ParseNode(buffer, off, off + len, subgraphs, constants));
            else if (field == GraphInitializer) inits.Add(ParseTensor(buffer, off, off + len));
        }
    }

    private static OnnxNode ParseNode(ReadOnlySpan<byte> buffer, int start, int end,
        List<OnnxSubgraph> subgraphs, List<OnnxTensor>? constants)
    {
        List<string> inputs = [], outputs = [];
        string name = "", opType = "";
        List<(int Off, int Len)> attributes = [];
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case NodeInput when wire == ProtoReader.WireLengthDelimited: inputs.Add(r.ReadString()); break;
                case NodeOutput when wire == ProtoReader.WireLengthDelimited: outputs.Add(r.ReadString()); break;
                case NodeName when wire == ProtoReader.WireLengthDelimited: name = r.ReadString(); break;
                case NodeOpType when wire == ProtoReader.WireLengthDelimited: opType = r.ReadString(); break;
                case NodeAttribute when wire == ProtoReader.WireLengthDelimited: attributes.Add(r.ReadLengthDelimited()); break;
                default: r.Skip(wire); break;
            }
        }
        // Attributes are parsed after the node's own fields because a Constant's value only means anything once
        // the op type is known, and protobuf does not promise field order.
        foreach ((int off, int len) in attributes)
        {
            ParseAttribute(buffer, off, off + len, name, opType, subgraphs, constants);
        }
        return new OnnxNode { OpType = opType, Name = name, Inputs = [.. inputs], Outputs = [.. outputs] };
    }

    private static void ParseAttribute(ReadOnlySpan<byte> buffer, int start, int end,
        string nodeName, string opType, List<OnnxSubgraph> subgraphs, List<OnnxTensor>? constants)
    {
        string attrName = "";
        (int Off, int Len)? tensor = null;
        (int Off, int Len)? graph = null;
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case AttrName when wire == ProtoReader.WireLengthDelimited: attrName = r.ReadString(); break;
                case AttrTensor when wire == ProtoReader.WireLengthDelimited: tensor = r.ReadLengthDelimited(); break;
                case AttrGraph when wire == ProtoReader.WireLengthDelimited: graph = r.ReadLengthDelimited(); break;
                default: r.Skip(wire); break;
            }
        }
        if (tensor is not null && constants is not null && opType == "Constant" && attrName == "value")
        {
            constants.Add(ParseTensor(buffer, tensor.Value.Off, tensor.Value.Off + tensor.Value.Len));
        }
        if (graph is null) return;

        List<OnnxTensor> nestedInits = [], nestedConstants = [];
        List<OnnxNode> nestedNodes = [];
        ParseGraph(buffer, graph.Value.Off, graph.Value.Off + graph.Value.Len,
            nestedInits, nestedNodes, subgraphs, nestedConstants);
        subgraphs.Add(new OnnxSubgraph
        {
            AttributeName = attrName,
            NodeName = nodeName,
            Initializers = nestedInits,
            Constants = nestedConstants,
            Nodes = nestedNodes,
        });
    }

    private static OnnxTensor ParseTensor(ReadOnlySpan<byte> buffer, int start, int end)
    {
        List<long> dims = [];
        int dataType = 0, rawOff = 0, rawLen = 0;
        string name = "";
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                // repeated int64 dims — pytorch exports unpacked (one varint per dim); also accept packed.
                case TensorDims when wire == ProtoReader.WireVarint: dims.Add((long)r.ReadVarint()); break;
                case TensorDims when wire == ProtoReader.WireLengthDelimited:
                {
                    (int po, int pl) = r.ReadLengthDelimited();
                    ProtoReader packed = new(buffer, po, po + pl);
                    while (packed.HasMore) dims.Add((long)packed.ReadVarint());
                    break;
                }
                case TensorDataType when wire == ProtoReader.WireVarint: dataType = (int)r.ReadVarint(); break;
                case TensorName when wire == ProtoReader.WireLengthDelimited: name = r.ReadString(); break;
                case TensorRawData when wire == ProtoReader.WireLengthDelimited:
                    (rawOff, rawLen) = r.ReadLengthDelimited();
                    break;
                default: r.Skip(wire); break;
            }
        }
        return new OnnxTensor
        {
            Name = name,
            Dims = [.. dims],
            DataType = dataType,
            RawOffset = rawOff,
            RawLength = rawLen,
        };
    }
}
