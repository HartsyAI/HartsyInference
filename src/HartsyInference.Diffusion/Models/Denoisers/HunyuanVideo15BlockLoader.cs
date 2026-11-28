using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Where a block's device copies go: the upload and the release. The loader is its only caller, so a failure in either is handled in one place.</summary>
internal interface IBlockPlacement
{
    void Preload(IEnumerable<Tensor> weights);

    void Free(IEnumerable<Tensor> weights);

    void Sync();

    /// <summary>Releases a host copy the loader owns. Defaults to the tensor's own dispose.</summary>
    void DisposeHost(Tensor tensor) => tensor.Dispose();
}

/// <summary>The placement on a backend: <see cref="IBackend.PreloadWeights"/>, <see cref="IBackend.FreeWeights"/> and <see cref="IBackend.Sync"/>.</summary>
internal sealed class BackendBlockPlacement(IBackend backend) : IBlockPlacement
{
    public void Preload(IEnumerable<Tensor> weights) => backend.PreloadWeights(weights);

    public void Free(IEnumerable<Tensor> weights) => backend.FreeWeights(weights);

    public void Sync() => backend.Sync();
}

/// <summary>The single owner of the released MovieRoll DiT's block weights when they stream: it loads a block's host weights from the checkpoint as the
/// forward reaches it, uploads them to the device, and on the block's release frees the device copy and then the host copies. The resident load
/// converts and casts every block at once and holds the whole F16 DiT on the host (about 15.5 GiB for the released file); this holds one block.
/// <para>Each block goes through the same converter and the same F32 to F16 cast as the resident load, so the forward's output is bit-identical.
/// The shared (top-level) weights are converted and cast once, in <see cref="LoadShared"/>; the caller uploads them for the run.</para>
/// <para>Failures: a load that throws at any step, including the upload, leaves no block loaded and releases what it created. A release that throws
/// still clears the loaded block and its host copies. Releasing a block that is not loaded does nothing.</para>
/// <para>Wiring: <c>dit.LoadBlock = loader.LoadBlock</c> and <c>dit.UnloadBlock = loader.UnloadBlock</c>. The DiT calls them around each block's forward,
/// so each block is loaded and released once per forward. No streaming scope may manage these blocks (<see cref="HunyuanVideo15Dit.StreamsBlockWeights"/>).</para>
/// <para>File reads: a block's F32 tensors are read from the checkpoint with positional reads (<see cref="MmapHandle.ReadAt"/>) into owned buffers, not
/// through the mapping, so the file's pages stay in the system cache rather than the process's working set. A tensor that is not registered as a file
/// view is used as it is.</para></summary>
public sealed class HunyuanVideo15BlockLoader : IDisposable
{
    private readonly HunyuanVideo15Dit _dit;
    private readonly HunyuanVideo15Config _config;
    private readonly IBlockPlacement _placement;
    private readonly IReadOnlyDictionary<string, Tensor> _source;
    private readonly IReadOnlyList<DType> _castSources;
    private readonly HashSet<Tensor> _mapped;
    private readonly bool _readBlocksFromFile;
    private readonly List<Tensor> _shared = [];
    private List<Tensor> _blockOwned = [];
    private int _loaded = -1;
    private bool _disposed;

    /// <summary>Block loads that completed. Each forward loads every block once, so this counts forwards times blocks.</summary>
    public int BlockLoads { get; private set; }

    /// <summary>Block releases so far. Equal to <see cref="BlockLoads"/> whenever no block is loaded.</summary>
    public int BlockReleases { get; private set; }

    /// <summary>The block now loaded, or -1 when none is.</summary>
    public int LoadedBlock => _loaded;

    /// <summary>Host tensors held for the loaded block; 0 when none is loaded. Shared weights are not counted.</summary>
    public int LoadedBlockTensorCount => _blockOwned.Count;

    /// <param name="source">The checkpoint's tensors as mapped views (the container's <c>Weights</c>); they are read, never released.</param>
    /// <param name="castSources">The dtypes the F32 to F16 cast applies to (the same list the resident load uses for the backend).</param>
    /// <param name="readBlocksFromFile">True reads each F32 block tensor with positional reads of its file; false uses the views.</param>
    public HunyuanVideo15BlockLoader(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IBackend backend, IReadOnlyDictionary<string, Tensor> source,
        IReadOnlyList<DType> castSources, bool readBlocksFromFile = true)
        : this(dit, config, new BackendBlockPlacement(backend ?? throw new ArgumentNullException(nameof(backend))), source, castSources, readBlocksFromFile)
    {
    }

    internal HunyuanVideo15BlockLoader(HunyuanVideo15Dit dit, HunyuanVideo15Config config, IBlockPlacement placement,
        IReadOnlyDictionary<string, Tensor> source, IReadOnlyList<DType> castSources, bool readBlocksFromFile)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _placement = placement ?? throw new ArgumentNullException(nameof(placement));
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

    /// <summary>Converts, casts and loads block <paramref name="index"/>, then uploads it to the device. A block still loaded from an earlier forward is
    /// released first. If any step throws, the block is not loaded and everything this call created is released before the exception leaves.</summary>
    public void LoadBlock(int index)
    {
        ThrowIfDisposed();
        if ((uint)index >= (uint)_config.NumBlocks) throw new ArgumentOutOfRangeException(nameof(index));
        if (_loaded >= 0) UnloadBlock(_loaded);

        List<Tensor> created = [];
        try
        {
            string prefix = $"double_blocks.{index}.";
            Dictionary<string, Tensor> input = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in _source)
            {
                if (pair.Key.StartsWith(prefix, StringComparison.Ordinal))
                    input["double_blocks.0." + pair.Key[prefix.Length..]] = ReadBlockTensor(pair.Value, created);
                else if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal)) input[pair.Key] = pair.Value;
            }
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(input, numBlocks: 1, hidden: _config.HiddenSize);
            created.AddRange(converted.Values);
            Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> pair in converted)
            {
                if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                    blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
            }
            Dictionary<string, Tensor> cast = Cast(blockWeights);
            created.AddRange(cast.Values);

            // The block is recorded as loaded before the upload, so a failing upload is rolled back through the same path as a failing release.
            _blockOwned = Settle(created, cast);
            _loaded = index;
            _dit.LoadBlockWeights(index, cast);
            _placement.Preload(_dit.GetBlock(index).EnumerateWeights());
            BlockLoads++;
        }
        catch
        {
            RollBackLoad(index, created);
            throw;
        }
    }

    /// <summary>Releases block <paramref name="index"/>: the device copy first (after a sync, so no queued work reads it), then the host copies. Releasing
    /// a block that is not the loaded one does nothing, so a retry after a failed release is safe.</summary>
    public void UnloadBlock(int index)
    {
        ThrowIfDisposed();
        if (index != _loaded) return;
        Exception? failure = null;
        try
        {
            _placement.Sync();
            _placement.Free(_dit.GetBlock(index).EnumerateWeights());
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        // The state is cleared before any host copy is released, so a release that throws cannot leave the block loaded.
        List<Tensor> owned = _blockOwned;
        _blockOwned = [];
        _loaded = -1;
        BlockReleases++;
        DisposeHostCopies(owned, ref failure);
        if (failure is not null) ExceptionDispatchInfo.Capture(failure).Throw();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        Exception? failure = null;
        try
        {
            if (_loaded >= 0) UnloadBlock(_loaded);
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        // The shared copies are released even when the block's release failed.
        _disposed = true;
        List<Tensor> shared = [.. _shared];
        _shared.Clear();
        DisposeHostCopies(shared, ref failure);
        if (failure is not null) ExceptionDispatchInfo.Capture(failure).Throw();
    }

    /// <summary>Releases each host copy on its own: a copy that fails to release does not stop the others, and the first failure is reported.</summary>
    private void DisposeHostCopies(IEnumerable<Tensor> tensors, ref Exception? failure)
    {
        foreach (Tensor tensor in tensors)
        {
            try
            {
                _placement.DisposeHost(tensor);
            }
            catch (Exception ex)
            {
                failure ??= ex;
            }
        }
    }

    /// <summary>Undoes a load that threw: the device copies are freed on a best-effort basis, then every tensor the load created and kept is released.
    /// Mapped views are never released, and a second dispose is a no-op.</summary>
    private void RollBackLoad(int index, List<Tensor> created)
    {
        // The state is reset first, so nothing below can leave the block marked as loaded. Nothing below may throw: the load's own exception is the one
        // the caller sees.
        List<Tensor> owned = _blockOwned;
        _blockOwned = [];
        _loaded = -1;
        try
        {
            _placement.Free(_dit.GetBlock(index).EnumerateWeights());
        }
        catch
        {
            // The upload that failed may have placed nothing; a failing free here must not hide the load's own exception.
        }
        Exception? ignored = null;
        List<Tensor> held = [];
        foreach (Tensor tensor in created.Concat(owned))
        {
            if (!_mapped.Contains(tensor)) held.Add(tensor);
        }
        DisposeHostCopies(held, ref ignored);
    }

    /// <summary>The block tensor for a checkpoint view: an owned F32 copy read with positional reads when this loader reads from the file and the view is
    /// a registered file view; otherwise the view. The copy is added to <paramref name="created"/> before it is filled, so a failed read is released too.</summary>
    private Tensor ReadBlockTensor(Tensor view, List<Tensor> created)
    {
        if (!_readBlocksFromFile || view.DType != DType.F32 || !TensorFileExtents.TryGet(view, out MmapHandle? owner, out long fileOffset)) return view;
        Tensor tensor = new(view.Shape, DType.F32);
        created.Add(tensor);
        owner.ReadAt(fileOffset, MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
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
