namespace HartsyInference.ModelAssets.Onnx;

/// <summary>A parsed ONNX <c>TensorProto</c> initializer: its name, shape, element type, and the absolute
/// byte range of its <c>raw_data</c> blob in the source buffer (so it can be copied straight into native
/// memory with no intermediate allocation).</summary>
public sealed class OnnxTensor
{
    public required string Name { get; init; }
    public required long[] Dims { get; init; }
    /// <summary>ONNX <c>TensorProto.DataType</c> (1 = FLOAT, 10 = FLOAT16, 7 = INT64, 6 = INT32, …).</summary>
    public required int DataType { get; init; }
    public required int RawOffset { get; init; }
    public required int RawLength { get; init; }
}

/// <summary>A parsed ONNX <c>NodeProto</c> in graph (topological) order: its op type, inputs, outputs, and
/// optional name. Used to bind anonymized initializers (<c>onnx::MatMul_NNNN</c>) to logical layers by walking
/// which node consumes them.</summary>
public sealed class OnnxNode
{
    public required string OpType { get; init; }
    public required string Name { get; init; }
    public required string[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
}

/// <summary>The subset of an ONNX model the engine needs: the initializer tensors (weights) and the node list
/// in graph order. Parsed by walking the protobuf wire format directly (see <see cref="ProtoReader"/>); only
/// <c>ModelProto.graph</c> → {<c>node</c>, <c>initializer</c>} are decoded, every other field is skipped.</summary>
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
    // TensorProto field numbers.
    private const int TensorDims = 1;
    private const int TensorDataType = 2;
    private const int TensorName = 8;
    private const int TensorRawData = 9;

    public IReadOnlyList<OnnxTensor> Initializers { get; }
    public IReadOnlyList<OnnxNode> Nodes { get; }

    private OnnxModel(List<OnnxTensor> initializers, List<OnnxNode> nodes)
    {
        Initializers = initializers;
        Nodes = nodes;
    }

    /// <summary>Parses the <c>ModelProto</c> in <paramref name="buffer"/> (the full <c>.onnx</c> file bytes).</summary>
    public static OnnxModel Parse(ReadOnlySpan<byte> buffer)
    {
        List<OnnxTensor> inits = [];
        List<OnnxNode> nodes = [];
        ProtoReader r = new(buffer);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (field == ModelGraph && wire == ProtoReader.WireLengthDelimited)
            {
                (int off, int len) = r.ReadLengthDelimited();
                ParseGraph(buffer, off, off + len, inits, nodes);
            }
            else
            {
                r.Skip(wire);
            }
        }
        return new OnnxModel(inits, nodes);
    }

    private static void ParseGraph(ReadOnlySpan<byte> buffer, int start, int end,
        List<OnnxTensor> inits, List<OnnxNode> nodes)
    {
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            if (wire != ProtoReader.WireLengthDelimited) { r.Skip(wire); continue; }
            (int off, int len) = r.ReadLengthDelimited();
            if (field == GraphNode) nodes.Add(ParseNode(buffer, off, off + len));
            else if (field == GraphInitializer) inits.Add(ParseTensor(buffer, off, off + len));
        }
    }

    private static OnnxNode ParseNode(ReadOnlySpan<byte> buffer, int start, int end)
    {
        List<string> inputs = [], outputs = [];
        string name = "", opType = "";
        ProtoReader r = new(buffer, start, end);
        while (r.TryReadTag(out int field, out int wire))
        {
            switch (field)
            {
                case NodeInput when wire == ProtoReader.WireLengthDelimited: inputs.Add(r.ReadString()); break;
                case NodeOutput when wire == ProtoReader.WireLengthDelimited: outputs.Add(r.ReadString()); break;
                case NodeName when wire == ProtoReader.WireLengthDelimited: name = r.ReadString(); break;
                case NodeOpType when wire == ProtoReader.WireLengthDelimited: opType = r.ReadString(); break;
                default: r.Skip(wire); break;
            }
        }
        return new OnnxNode { OpType = opType, Name = name, Inputs = [.. inputs], Outputs = [.. outputs] };
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
