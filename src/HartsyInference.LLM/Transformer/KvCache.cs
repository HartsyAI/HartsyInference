using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Device-resident per-layer K/V cache for the GPU-resident decode loop: each layer's K/V is a backend activation tensor grown on-device with <see cref="IBackend.Concat"/> along the sequence axis, which on CUDA is a device-to-device copy so the cache never forces a device-to-host sync.</summary>
/// <remarks>
/// <para>Layout per layer: <c>[batch, num_kv_heads, currentLength, head_dim]</c>. The stored tensor is the populated prefix (no padding), handed straight to the GQA repeat then SDPA. Growth is O(n) per step (a fresh concat), i.e. O(n²) total — fine for the current scope; a fixed-buffer in-place append is the later optimization.</para>
/// <para>Usage: read <see cref="CurrentLength"/> for the position offset before the layer loop, <see cref="Append"/> each layer's new K/V during the loop, then <see cref="AdvanceLength"/> once after all layers.</para>
/// </remarks>
public sealed class KvCache : IKvCache, IDisposable
{
    private readonly Tensor?[] _k;
    private readonly Tensor?[] _v;
    private int _currentLength;
    private int _disposed;

    /// <summary>Number of layers this cache stores.</summary>
    public int NumLayers => _k.Length;

    /// <summary>Batch size (1 in the current scope).</summary>
    public int BatchSize { get; }

    /// <summary>Number of key/value heads per layer.</summary>
    public int NumKvHeads { get; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; }

    /// <summary>Tokens currently stored. <c>0</c> after construction and after <see cref="Reset"/>.</summary>
    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Creates per-layer device caches for a model with the given K/V geometry.</summary>
    public KvCache(int numLayers, int batch, int numKvHeads, int headDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be > 0");
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch must be > 0");
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads), numKvHeads, "numKvHeads must be > 0");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "headDim must be > 0");

        BatchSize = batch;
        NumKvHeads = numKvHeads;
        HeadDim = headDim;
        _k = new Tensor?[numLayers];
        _v = new Tensor?[numLayers];
    }

    /// <summary>Appends a new K/V block <c>[B, num_kv_heads, tNew, head_dim]</c> for <paramref name="layer"/> on-device; the cache takes a fresh resident copy (concat), so the caller still owns and may dispose the inputs. Does not change <see cref="CurrentLength"/> — call <see cref="AdvanceLength"/> after all layers.</summary>
    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
    {
        ThrowIfDisposed();
        _k[layer] = Grow(backend, _k[layer], newK);
        _v[layer] = Grow(backend, _v[layer], newV);
    }

    private Tensor Grow(IBackend backend, Tensor? existing, Tensor add)
    {
        int tNew = (int)add.Shape[2];
        if (existing is null)
        {
            Tensor copy = new(new TensorShape(BatchSize, NumKvHeads, tNew, HeadDim), DType.F32);
            backend.Concat(copy, [add], dim: 2);
            return copy;
        }
        int prevLen = (int)existing.Shape[2];
        Tensor grown = new(new TensorShape(BatchSize, NumKvHeads, prevLen + tNew, HeadDim), DType.F32);
        backend.Concat(grown, [existing, add], dim: 2);
        existing.Dispose();
        return grown;
    }

    /// <summary>The layer's resident K prefix <c>[B, num_kv_heads, len, head_dim]</c>.</summary>
    public Tensor KeyPrefix(int layer)
    {
        ThrowIfDisposed();
        return _k[layer] ?? throw new InvalidOperationException($"Layer {layer} K not populated.");
    }

    /// <summary>The layer's resident V prefix <c>[B, num_kv_heads, len, head_dim]</c>.</summary>
    public Tensor ValuePrefix(int layer)
    {
        ThrowIfDisposed();
        return _v[layer] ?? throw new InvalidOperationException($"Layer {layer} V not populated.");
    }

    /// <summary>Advances the shared position counter by <paramref name="by"/> tokens; call once per step, after every layer has appended.</summary>
    public void AdvanceLength(int by)
    {
        ThrowIfDisposed();
        if (by < 0) throw new ArgumentOutOfRangeException(nameof(by), by, "by must be >= 0");
        _currentLength += by;
    }

    /// <summary>Drops all stored K/V and resets the position counter (cheap reuse across generations).</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _k.Length; i++)
        {
            _k[i]?.Dispose(); _k[i] = null;
            _v[i]?.Dispose(); _v[i] = null;
        }
        _currentLength = 0;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(KvCache));
    }

    /// <summary>Frees all stored device K/V.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _k.Length; i++)
        {
            _k[i]?.Dispose();
            _v[i]?.Dispose();
        }
    }
}
