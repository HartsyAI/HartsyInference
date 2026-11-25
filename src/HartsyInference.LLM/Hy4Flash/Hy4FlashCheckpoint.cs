using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>A Hy4 Flash checkpoint directory as an <see cref="IHy4FlashWeightSource"/>: reads through <see cref="ShardedSafeTensorSet"/>, decodes block-scaled
/// e4m3 weights (a tensor whose companion <c>&lt;name&gt;_scale_inv</c> holds the 128x128 BF16 or F32 scale grid) to F32, and converts BF16 or F16 to F32.</summary>
/// <remarks>Routed-expert stacks and the BF16 projections are returned as F32 copies, which is what the host reference computes in. That copies the whole model into host memory,
/// so it holds for the tiny checkpoints only; the released 220B checkpoint needs the device-resident FP8 path, which is not part of this reference. The embedding table and the Engram
/// tables are returned as mapped views in their stored dtype, read by row.</remarks>
public sealed class Hy4FlashCheckpoint : IHy4FlashWeightSource, IDisposable
{
    private readonly ShardedSafeTensorSet _set;
    private int _disposed;

    private Hy4FlashCheckpoint(Hy4FlashConfig config, ShardedSafeTensorSet set)
    {
        Config = config;
        _set = set;
    }

    /// <summary>The config read from the directory's <c>config.json</c>.</summary>
    public Hy4FlashConfig Config { get; }

    /// <summary>Opens <paramref name="directory"/>: its <c>config.json</c> and either the sharded index or every <c>.safetensors</c> file in it.</summary>
    /// <exception cref="FileNotFoundException">No <c>config.json</c>.</exception>
    public static Hy4FlashCheckpoint Open(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(directory, "config.json"));
        ShardedSafeTensorSet set = File.Exists(Path.Combine(directory, "model.safetensors.index.json"))
            ? ShardedSafeTensorSet.OpenIndex(directory)
            : ShardedSafeTensorSet.OpenFiles([.. Directory.GetFiles(directory, "*.safetensors").Order(StringComparer.Ordinal)]);
        return new Hy4FlashCheckpoint(config, set);
    }

    /// <inheritdoc />
    public bool Contains(string name) => _set.Inventory.ContainsKey(name);

    /// <inheritdoc />
    public Tensor Read(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (!_set.Inventory.TryGetValue(name, out TensorLocation? location))
            throw new KeyNotFoundException($"The Hy4 Flash checkpoint has no tensor '{name}'.");
        if (location.DType == DType.F8E4M3) return Dequantize(name, location);
        if (location.DType == DType.F16)
            throw new InvalidDataException($"Tensor '{name}' is F16; the Hy4 Flash checkpoint stores BF16 and F8 only, so an F16 tensor is refused rather than converted.");
        if (IsLargeTable(name)) return _set.GetTensor(name);
        Tensor t = _set.ReadTensor(name);
        if (t.DType != DType.BF16) return t;
        using (t) return ProjectionOps.EnsureF32(t);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _set.Dispose();
    }

    private static bool IsLargeTable(string name) =>
        name == "model.embed_tokens.weight" || name.Contains(".engram_layer.multi_head_embedding.", StringComparison.Ordinal);

    private unsafe Tensor Dequantize(string name, TensorLocation location)
    {
        string scaleName = Hy4FlashWeightMap.ScaleKey(name);
        if (!_set.Inventory.ContainsKey(scaleName))
            throw new InvalidDataException($"FP8 tensor '{name}' has no scale grid '{scaleName}'.");
        float[] scale;
        using (Tensor scaleTensor = Read(scaleName)) scale = scaleTensor.AsReadOnlySpan<float>().ToArray();

        long[] dims = new long[location.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = location.Shape[i];
        using Tensor raw = _set.ReadTensor(name);
        ReadOnlySpan<byte> codes = new(raw.DataPointer, checked((int)raw.ElementCount));
        Tensor output = new(new TensorShape(dims), DType.F32);
        try
        {
            Hy4FlashFp8.DequantizeBlocks(codes, dims, scale, output.AsSpan<float>());
        }
        catch
        {
            output.Dispose();
            throw;
        }
        return output;
    }
}
