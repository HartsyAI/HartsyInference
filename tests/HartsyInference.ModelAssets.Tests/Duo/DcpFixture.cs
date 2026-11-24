using System.IO.Compression;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>Builds a distributed-checkpoint folder in the layout PyTorch writes: one standalone <c>torch.save</c> ZIP per tensor concatenated in <c>__0_0.distcp</c>, and a <c>.metadata</c> pickle with the tensor table and each tensor's byte range.</summary>
internal static class DcpFixture
{
    public readonly record struct Tensor1(DType DType, long[] Shape, byte[] Bytes);

    public static Tensor1 F32(long[] shape, params float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return new Tensor1(DType.F32, shape, bytes);
    }

    public static Tensor1 Bf16(long[] shape, params float[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
            BitConverter.GetBytes(HartsyInference.Core.Numerics.BFloat16Bits.FromFloat(values[i])).CopyTo(bytes, i * 2);
        return new Tensor1(DType.BF16, shape, bytes);
    }

    /// <summary>Writes the folder. <paramref name="chunkOverride"/> turns the named tensor's chunk list into one half-size chunk, to exercise the sharded-tensor refusal.</summary>
    public static void Write(string folder, IReadOnlyDictionary<string, Tensor1> tensors, string? chunkOverride = null, string relativePath = "__0_0.distcp")
    {
        Directory.CreateDirectory(folder);
        using MemoryStream distcp = new();
        Dictionary<string, (long Offset, long Length)> ranges = [];
        foreach ((string key, Tensor1 tensor) in tensors)
        {
            byte[] zip = TorchZip(tensor);
            ranges[key] = (distcp.Position, zip.Length);
            distcp.Write(zip);
        }
        File.WriteAllBytes(Path.Combine(folder, "__0_0.distcp"), distcp.ToArray());
        File.WriteAllBytes(Path.Combine(folder, ".metadata"), Metadata(tensors, ranges, chunkOverride, relativePath));
    }

    public static byte[] TorchZip(Tensor1 tensor)
    {
        PickleBuilder pickle = new(proto4: false);
        string storage = tensor.DType == DType.F32 ? "FloatStorage" : tensor.DType == DType.BF16 ? "BFloat16Storage" : throw new NotSupportedException();
        long elements = tensor.Bytes.Length / tensor.DType.SizeInBytes;
        long[] stride = new long[tensor.Shape.Length];
        long acc = 1;
        for (int i = stride.Length - 1; i >= 0; i--) { stride[i] = acc; acc *= tensor.Shape[i]; }

        pickle.Global("torch._utils", "_rebuild_tensor_v2");
        pickle.Tuple(
            b => b.Tuple(
                x => x.Str("storage"), x => x.Global("torch", storage), x => x.Str("0"), x => x.Str("cpu"), x => x.Int(elements))
                .Op((byte)'Q'),
            b => b.Int(0),
            b => b.Tuple([.. tensor.Shape.Select<long, Action<PickleBuilder>>(d => x => x.Int(d))]),
            b => b.Tuple([.. stride.Select<long, Action<PickleBuilder>>(d => x => x.Int(d))]),
            b => b.Bool(false),
            b => b.Global("collections", "OrderedDict").Op((byte)')').Op((byte)'R'));
        pickle.Op((byte)'R');

        using MemoryStream stream = new();
        using (ZipArchive zip = new(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            Add(zip, "archive/data.pkl", pickle.Finish());
            Add(zip, "archive/data/0", tensor.Bytes);
            Add(zip, "archive/version", "3\n"u8.ToArray());
        }
        return stream.ToArray();
    }

    private static void Add(ZipArchive zip, string name, byte[] bytes)
    {
        ZipArchiveEntry entry = zip.CreateEntry(name, CompressionLevel.NoCompression);
        using Stream s = entry.Open();
        s.Write(bytes);
    }

    private static byte[] Metadata(IReadOnlyDictionary<string, Tensor1> tensors, Dictionary<string, (long Offset, long Length)> ranges,
        string? chunkOverride, string relativePath)
    {
        const string MetadataModule = "torch.distributed.checkpoint.metadata";
        PickleBuilder p = new();
        p.Object(MetadataModule, "Metadata",
            ("state_dict_metadata", b => b.Dict(tensors.Select<KeyValuePair<string, Tensor1>, (Action<PickleBuilder>, Action<PickleBuilder>)>(
                t => (x => x.Str(t.Key), x => TensorMetadata(x, t.Value, t.Key == chunkOverride))))),
            ("planner_data", b => b.None()),
            ("storage_data", b => b.Dict(tensors.Keys.Select<string, (Action<PickleBuilder>, Action<PickleBuilder>)>(
                key => (
                    x => x.Object(MetadataModule, "MetadataIndex",
                        ("fqn", y => y.Str(key)), ("offset", y => y.TorchSize(new long[tensors[key].Shape.Length])), ("index", y => y.None())),
                    x => x.Object("torch.distributed.checkpoint.filesystem", "_StorageInfo",
                        ("relative_path", y => y.Str(relativePath)), ("offset", y => y.Int(ranges[key].Offset)),
                        ("length", y => y.Int(ranges[key].Length)), ("transform_descriptors", y => y.None())))))),
            ("storage_meta", b => b.None()));
        return p.Finish();
    }

    private static void TensorMetadata(PickleBuilder b, Tensor1 tensor, bool halfChunk)
    {
        long[] zeros = new long[tensor.Shape.Length];
        long[] sizes = halfChunk ? tensor.Shape.Select(d => Math.Max(1, d / 2)).ToArray() : tensor.Shape;
        string dtype = tensor.DType == DType.F32 ? "float32" : "bfloat16";
        b.Object("torch.distributed.checkpoint.metadata", "TensorStorageMetadata",
            // TensorProperties.__getstate__ returns (dtype, layout, requires_grad, _MEM_FORMAT_ENCODING member, pin_memory); an Enum member pickles as cls(value).
            ("properties", x => x.ObjectWithTupleState("torch.distributed.checkpoint.metadata", "TensorProperties",
                y => y.Global("torch", dtype), y => y.Global("torch", "strided"), y => y.Bool(false),
                y => y.Global("torch.distributed.checkpoint.metadata", "_MEM_FORMAT_ENCODING").Tuple(z => z.Int(0)).Op((byte)'R'),
                y => y.Bool(false))),
            ("size", x => x.TorchSize(tensor.Shape)),
            ("chunks", x => x.List([y => y.Object("torch.distributed.checkpoint.metadata", "ChunkStorageMetadata",
                ("offsets", z => z.TorchSize(zeros)), ("sizes", z => z.TorchSize(sizes)))])));
    }
}
