using HartsyInference.Core.IO;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Kimi;

/// <summary>Kimi K3.1 tensors read from a safetensors checkpoint directory through its <c>model.safetensors.index.json</c>. A whole tensor is a
/// borrowed view of its shard's mapping (the view keeps the mapping alive), or a copy when its shard is pread-only; a row or column range
/// is pread from the shard, so a rank reads only its share. Every tensor keeps the dtype the checkpoint stores; the loader converts.</summary>
public sealed class KimiK31CheckpointWeights : IKimiK31RankWeightSource, IDisposable
{
    private ShardedSafeTensorSet? _set;

    private KimiK31CheckpointWeights(ShardedSafeTensorSet set) => _set = set;

    /// <summary>Opens the checkpoint's shards.</summary>
    /// <exception cref="HartsyInference.Core.HartsyInferenceException">The index is missing or a shard file is absent.</exception>
    public static KimiK31CheckpointWeights Open(string directory) => new(ShardedSafeTensorSet.OpenIndex(directory));

    /// <inheritdoc/>
    public bool Has(string name) => Set.Inventory.ContainsKey(name);

    /// <inheritdoc/>
    public Tensor Read(string name)
    {
        ShardedSafeTensorSet set = Set;
        bool pread = set.Inventory.TryGetValue(name, out TensorLocation? location) && location.Shard.PreadOnly;
        return pread ? set.ReadTensor(name) : set.GetTensor(name);
    }

    /// <inheritdoc/>
    public unsafe Tensor ReadSlice(string name, int dim, long start, long count)
    {
        ShardedSafeTensorSet set = Set;
        if (!set.Inventory.TryGetValue(name, out TensorLocation? location))
            throw new KeyNotFoundException($"Tensor '{name}' is not in the checkpoint.");
        TensorShape shape = location.Shape;
        if ((uint)dim >= (uint)shape.Rank) throw new ArgumentOutOfRangeException(nameof(dim), dim, $"'{name}' has rank {shape.Rank}.");
        if (dim == 1 && shape.Rank != 2) throw new ArgumentOutOfRangeException(nameof(dim), dim, "Column ranges are read for rank-2 tensors only.");
        if (start < 0 || count < 1 || start + count > shape[dim])
            throw new ArgumentOutOfRangeException(nameof(count), count, $"[{start}, {start + count}) lies outside '{name}' dimension {dim} of {shape[dim]}.");

        long[] dims = new long[shape.Rank];
        for (int d = 0; d < dims.Length; d++) dims[d] = shape[d];
        dims[dim] = count;
        Tensor slice = new(new TensorShape(dims), location.DType);
        IWeightByteSource source = set.GetByteSource(location.Shard);
        if (dim == 0)
        {
            long rowBytes = location.ByteLength / shape[0];
            source.ReadAt(location.FileOffset + start * rowBytes, new Span<byte>(slice.DataPointer, checked((int)(count * rowBytes))));
            return slice;
        }
        long fullRowBytes = location.DType.ComputeByteCount(shape[1]);
        long sliceRowBytes = location.DType.ComputeByteCount(count);
        long startBytes = location.DType.ComputeByteCount(start);
        for (long row = 0; row < shape[0]; row++)
        {
            byte* at = (byte*)slice.DataPointer + row * sliceRowBytes;
            source.ReadAt(location.FileOffset + row * fullRowBytes + startBytes, new Span<byte>(at, checked((int)sliceRowBytes)));
        }
        return slice;
    }

    private ShardedSafeTensorSet Set => _set ?? throw new ObjectDisposedException(nameof(KimiK31CheckpointWeights));

    /// <inheritdoc/>
    public void Dispose()
    {
        ShardedSafeTensorSet? set = Interlocked.Exchange(ref _set, null);
        set?.Dispose();
    }
}
