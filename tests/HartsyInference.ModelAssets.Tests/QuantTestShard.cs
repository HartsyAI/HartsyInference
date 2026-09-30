using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Builds a one-file, zero-filled safetensors set from (name, dtype, shape) triples; binder tests read headers only.</summary>
internal sealed class QuantTestShard : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"quant_shard_{Guid.NewGuid():N}");
    private readonly List<ShardTestFiles.Entry> _entries = new();
    private ShardedSafeTensorSet? _set;

    internal QuantTestShard Add(string name, string dtype, params long[] shape)
    {
        long elements = shape.Aggregate(1L, (a, b) => a * b);
        _entries.Add(new ShardTestFiles.Entry(name, dtype, shape, new byte[elements * ElementBytes(dtype)]));
        return this;
    }

    internal ShardedSafeTensorSet Open()
    {
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "a.safetensors");
        ShardTestFiles.WriteShard(path, _entries.ToArray());
        _set = ShardedSafeTensorSet.OpenFiles([path]);
        return _set;
    }

    private static int ElementBytes(string dtype) => dtype switch
    {
        "F8_E4M3" or "F8_E8M0" or "I8" or "U8" => 1,
        "BF16" or "F16" => 2,
        "F32" or "U32" or "I32" => 4,
        _ => throw new ArgumentException($"Unknown test dtype {dtype}."),
    };

    public void Dispose()
    {
        _set?.Dispose();
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }
}
