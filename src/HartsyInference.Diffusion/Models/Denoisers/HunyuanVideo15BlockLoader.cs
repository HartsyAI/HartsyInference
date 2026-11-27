using System.Runtime.InteropServices;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Loads the released MovieRoll DiT one block at a time, from the checkpoint, as the forward reaches each block. The resident load converts
/// and casts every block at once and holds the whole F16 DiT on the host (about 15.5 GiB of the released F32 checkpoint); this loader holds one
/// block's F16 weights, and the previous block's are released when the next block loads.
/// <para>Each block goes through the same converter and the same F32 to F16 cast as the resident load, so the weights, and the forward's output, are
/// bit-identical. The shared (top-level) weights are converted and cast once, in <see cref="LoadShared"/>.</para>
/// <para>Block reads: built with <see cref="FromCheckpoint"/>, each F32 block tensor is read into an owned buffer with ordinary file reads, not
/// through the mapped view, so the file's pages go to the system cache (reclaimable) rather than the process's working set. Built from a dictionary
/// of mapped views, the block tensors are the views.</para>
/// <para>Ownership: a tensor the reads, the converter or the cast creates is released unless the DiT keeps it; a mapped view is never released.
/// Wire it with <c>dit.LoadBlock = loader.LoadBlock</c> after <see cref="LoadShared"/>, and dispose it after the last forward.</para></summary>
public sealed class HunyuanVideo15BlockLoader : IDisposable
{
    private readonly HunyuanVideo15Dit _dit;
    private readonly HunyuanVideo15Config _config;
    private readonly IReadOnlyDictionary<string, Tensor> _source;
    private readonly IReadOnlyList<DType> _castSources;
    private readonly HashSet<Tensor> _mapped;
    private readonly IReadOnlyDictionary<string, SafeTensorDescriptor>? _descriptors;
    private readonly FileStream? _file;
    private readonly List<Tensor> _shared = [];
    private List<Tensor> _previous = [];
    private bool _disposed;

    /// <param name="source">The checkpoint's tensors as mapped views (the container's <c>Weights</c>); they are read, never released.</param>
    /// <param name="castSources">The dtypes the F32 to F16 cast applies to (the same list the resident load uses for the backend).</param>
    public HunyuanVideo15BlockLoader(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IReadOnlyDictionary<string, Tensor> source,
        IReadOnlyList<DType> castSources)
        : this(dit, config, source, castSources, path: null, descriptors: null)
    {
    }

    private HunyuanVideo15BlockLoader(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IReadOnlyDictionary<string, Tensor> source,
        IReadOnlyList<DType> castSources, string? path, IReadOnlyDictionary<string, SafeTensorDescriptor>? descriptors)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _castSources = castSources ?? throw new ArgumentNullException(nameof(castSources));
        _mapped = new HashSet<Tensor>(source.Values, ReferenceEqualityComparer.Instance);
        _descriptors = descriptors;
        if (path is not null && descriptors is not null)
            _file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1, FileOptions.SequentialScan);
    }

    /// <summary>A loader that reads each F32 block tensor from the checkpoint file into an owned buffer (see the class remarks).</summary>
    public static HunyuanVideo15BlockLoader FromCheckpoint(HunyuanVideo15Dit dit, HunyuanVideo15Config config, CheckpointSource checkpoint,
        IReadOnlyList<DType> castSources)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        return FromFile(dit, config, checkpoint.Weights, castSources, checkpoint.Path, checkpoint.Header.Descriptors);
    }

    /// <summary>The file-reading loader over the given mapped views; <paramref name="descriptors"/> are the file's tensor descriptors.</summary>
    internal static HunyuanVideo15BlockLoader FromFile(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IReadOnlyDictionary<string, Tensor> source,
        IReadOnlyList<DType> castSources, string path, IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors) =>
        new(dit, config, source, castSources, path, descriptors);

    /// <summary>Converts and casts the top-level weights and loads them into the DiT. Block 0 is read only because the converter takes its keys;
    /// its outputs are released.</summary>
    public void LoadShared()
    {
        ThrowIfDisposed();
        Dictionary<string, Tensor> input = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in _source)
        {
            if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal) || pair.Key.StartsWith("double_blocks.0.", StringComparison.Ordinal))
                input[pair.Key] = pair.Value;
        }
        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(input, numBlocks: 1, hidden: _config.HiddenSize);
        Dictionary<string, Tensor> shared = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in converted)
        {
            if (!pair.Key.StartsWith("transformer_blocks.", StringComparison.Ordinal)) shared[pair.Key] = pair.Value;
        }
        Dictionary<string, Tensor> cast = Cast(shared);
        _shared.AddRange(Settle(converted.Values.Concat(cast.Values), cast));
        _dit.LoadSharedWeights(cast);
    }

    /// <summary>Converts, casts and loads one block's weights, then releases the previous block's. Blocks are loaded in forward order on every forward.</summary>
    public void LoadBlock(int index)
    {
        ThrowIfDisposed();
        if ((uint)index >= (uint)_config.NumBlocks) throw new ArgumentOutOfRangeException(nameof(index));
        string prefix = $"double_blocks.{index}.";
        Dictionary<string, Tensor> input = new(StringComparer.Ordinal);
        List<Tensor> reads = [];
        foreach (KeyValuePair<string, Tensor> pair in _source)
        {
            if (pair.Key.StartsWith(prefix, StringComparison.Ordinal))
            {
                Tensor block = ReadBlockTensor(pair.Key, pair.Value, reads);
                input["double_blocks.0." + pair.Key[prefix.Length..]] = block;
            }
            else if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal)) input[pair.Key] = pair.Value;
        }
        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(input, numBlocks: 1, hidden: _config.HiddenSize);
        Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in converted)
        {
            if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
        }
        Dictionary<string, Tensor> cast = Cast(blockWeights);
        List<Tensor> owned = Settle(reads.Concat(converted.Values).Concat(cast.Values), cast);
        _dit.LoadBlockWeights(index, cast);
        foreach (Tensor old in _previous) old.Dispose();
        _previous = owned;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor tensor in _previous) tensor.Dispose();
        _previous.Clear();
        foreach (Tensor tensor in _shared) tensor.Dispose();
        _shared.Clear();
        _file?.Dispose();
    }

    /// <summary>The block tensor for <paramref name="name"/>: an owned F32 copy read from the file when this loader has the file, else the view.</summary>
    private Tensor ReadBlockTensor(string name, Tensor view, List<Tensor> reads)
    {
        if (_file is null || _descriptors is null || view.DType != DType.F32) return view;
        SafeTensorDescriptor descriptor = _descriptors[name];
        Tensor tensor = new(descriptor.Shape, descriptor.DType);
        byte[] buffer = new byte[descriptor.ByteLength];
        _file.Seek(descriptor.DataOffset, SeekOrigin.Begin);
        _file.ReadExactly(buffer);
        buffer.AsSpan().CopyTo(MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
        reads.Add(tensor);
        return tensor;
    }

    private Dictionary<string, Tensor> Cast(Dictionary<string, Tensor> weights)
    {
        // The rule of VaePrecisionHelper.CastWeights: a tensor is cast when its dtype is one of the sources and is not already the target.
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in weights)
            result[pair.Key] = pair.Value.DType != DType.F16 && _castSources.Contains(pair.Value.DType) ? pair.Value.CastTo(DType.F16) : pair.Value;
        return result;
    }

    /// <summary>Releases the tensors that were created here and are not kept, and returns the created tensors the DiT keeps. Mapped views are skipped.</summary>
    private List<Tensor> Settle(IEnumerable<Tensor> created, IReadOnlyDictionary<string, Tensor> kept)
    {
        HashSet<Tensor> keep = new(kept.Values, ReferenceEqualityComparer.Instance);
        List<Tensor> owned = [];
        foreach (Tensor tensor in new HashSet<Tensor>(created, ReferenceEqualityComparer.Instance))
        {
            if (_mapped.Contains(tensor)) continue;
            if (keep.Contains(tensor)) owned.Add(tensor);
            else tensor.Dispose();
        }
        return owned;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
