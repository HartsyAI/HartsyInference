using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The single owner of the released MovieRoll DiT's block weights when they stream: it loads a block's host weights from the checkpoint as the
/// forward reaches it, uploads them to the device, and on the block's release frees the device copy and then the host copies. The resident load
/// converts and casts every block at once and holds the whole F16 DiT on the host (about 15.5 GiB for the released file); this holds one block.
/// <para>Each block goes through the same converter and the same F32 to F16 cast as the resident load, so the forward's output is bit-identical.
/// The shared (top-level) weights are converted and cast once, in <see cref="LoadShared"/>; the caller uploads them for the run.</para>
/// <para>Wiring: <c>dit.LoadBlock = loader.LoadBlock</c> and <c>dit.UnloadBlock = loader.UnloadBlock</c>. The DiT calls them around each block's forward,
/// so each block is loaded and released once per forward. No streaming scope may manage these blocks (<see cref="HunyuanVideo15Dit.StreamsBlockWeights"/>).</para>
/// <para>File reads: a block's F32 tensors are read from the checkpoint with positional reads (<see cref="MmapHandle.ReadAt"/>) into owned buffers, not
/// through the mapping, so the file's pages stay in the system cache rather than the process's working set. A tensor that is not registered as a file
/// view is used as it is.</para></summary>
public sealed class HunyuanVideo15BlockLoader : IDisposable
{
    private readonly HunyuanVideo15Dit _dit;
    private readonly HunyuanVideo15Config _config;
    private readonly IBackend _backend;
    private readonly IReadOnlyDictionary<string, Tensor> _source;
    private readonly IReadOnlyList<DType> _castSources;
    private readonly HashSet<Tensor> _mapped;
    private readonly bool _readBlocksFromFile;
    private readonly List<Tensor> _shared = [];
    private List<Tensor> _blockOwned = [];
    private int _loaded = -1;
    private bool _disposed;

    /// <param name="source">The checkpoint's tensors as mapped views (the container's <c>Weights</c>); they are read, never released.</param>
    /// <param name="castSources">The dtypes the F32 to F16 cast applies to (the same list the resident load uses for the backend).</param>
    /// <param name="readBlocksFromFile">True reads each F32 block tensor with positional reads of its file; false uses the views.</param>
    public HunyuanVideo15BlockLoader(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IBackend backend, IReadOnlyDictionary<string, Tensor> source,
        IReadOnlyList<DType> castSources, bool readBlocksFromFile = true)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _castSources = castSources ?? throw new ArgumentNullException(nameof(castSources));
        _mapped = new HashSet<Tensor>(source.Values, ReferenceEqualityComparer.Instance);
        _readBlocksFromFile = readBlocksFromFile;
    }

    /// <summary>The loader for a checkpoint container: the block reads go to its file (see the class remarks).</summary>
    public static HunyuanVideo15BlockLoader FromCheckpoint(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IBackend backend,
        CheckpointSource checkpoint, IReadOnlyList<DType> castSources)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        return new HunyuanVideo15BlockLoader(dit, config, backend, checkpoint.Weights, castSources, readBlocksFromFile: true);
    }

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

    /// <summary>Converts, casts and loads block <paramref name="index"/>, then uploads it to the device. A block still loaded from an earlier forward that
    /// threw is released first.</summary>
    public void LoadBlock(int index)
    {
        ThrowIfDisposed();
        if ((uint)index >= (uint)_config.NumBlocks) throw new ArgumentOutOfRangeException(nameof(index));
        if (_loaded >= 0) UnloadBlock(_loaded);

        string prefix = $"double_blocks.{index}.";
        Dictionary<string, Tensor> input = new(StringComparer.Ordinal);
        List<Tensor> reads = [];
        foreach (KeyValuePair<string, Tensor> pair in _source)
        {
            if (pair.Key.StartsWith(prefix, StringComparison.Ordinal))
                input["double_blocks.0." + pair.Key[prefix.Length..]] = ReadBlockTensor(pair.Value, reads);
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
        _blockOwned = Settle(reads.Concat(converted.Values).Concat(cast.Values), cast);
        _dit.LoadBlockWeights(index, cast);
        _loaded = index;
        _backend.PreloadWeights(_dit.GetBlock(index).EnumerateWeights());
    }

    /// <summary>Releases block <paramref name="index"/>: the device copy first (after a sync, so no queued work reads it), then the host copies.</summary>
    public void UnloadBlock(int index)
    {
        ThrowIfDisposed();
        if (index != _loaded) throw new InvalidOperationException($"Block {index} is not the loaded block ({_loaded}).");
        _backend.Sync();
        _backend.FreeWeights(_dit.GetBlock(index).EnumerateWeights());
        foreach (Tensor tensor in _blockOwned) tensor.Dispose();
        _blockOwned = [];
        _loaded = -1;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_loaded >= 0) UnloadBlock(_loaded);
        _disposed = true;
        foreach (Tensor tensor in _shared) tensor.Dispose();
        _shared.Clear();
    }

    /// <summary>The block tensor for a checkpoint view: an owned F32 copy read with positional reads when this loader reads from the file and the view is
    /// a registered file view; otherwise the view.</summary>
    private Tensor ReadBlockTensor(Tensor view, List<Tensor> reads)
    {
        if (!_readBlocksFromFile || view.DType != DType.F32 || !TensorFileExtents.TryGet(view, out MmapHandle? owner, out long fileOffset)) return view;
        Tensor tensor = new(view.Shape, DType.F32);
        owner.ReadAt(fileOffset, MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
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
