using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Streaming;

/// <summary>Device-resident per-layer K/V cache for the GPU-resident autoregressive decode path
/// (M0 LLM spike). Unlike <see cref="StreamingKvCache"/> — which stores host F32 buffers and is
/// written via CPU memcpy — this cache keeps each layer's K/V as a backend activation tensor and
/// grows it on-device with <see cref="IBackend.Concat"/> along the sequence axis. On CUDA the
/// concat is a device-to-device copy, so the cache never triggers a device-to-host sync and the
/// whole decode loop stays GPU-resident.
///
/// <para>Layout per layer: <c>[batch, num_kv_heads, currentLength, head_dim]</c>. The stored tensor
/// IS the populated prefix (no padding), so it can be handed straight to SDPA after the GQA repeat.
/// Growth is O(n) per step (a fresh concat allocation), i.e. O(n²) total — acceptable for the spike's
/// short generations; a fixed-buffer + in-place append kernel is the M1+ optimization.</para>
///
/// <para>Usage mirrors <see cref="StreamingKvCache"/>: read <see cref="CurrentLength"/> for the
/// position offset before the layer loop, <see cref="Append"/> each layer's new K/V during the loop,
/// then <see cref="AdvanceLength"/> once after all layers.</para></summary>
public sealed class DeviceKvCache : IDisposable
{
    private readonly Tensor?[] _k;
    private readonly Tensor?[] _v;
    private int _currentLength;
    private int _disposed;

    public int NumLayers => _k.Length;
    public int BatchSize { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }

    /// <summary>Tokens currently stored. <c>0</c> after construction and after <see cref="Reset"/>.</summary>
    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    public DeviceKvCache(int numLayers, int batch, int numKvHeads, int headDim)
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

    /// <summary>Appends a new K/V block for <paramref name="layer"/> on-device. <paramref name="newK"/>
    /// and <paramref name="newV"/> are <c>[B, num_kv_heads, tNew, head_dim]</c>. The cache takes a fresh
    /// resident copy (concat), so the caller still owns and may dispose <paramref name="newK"/>/<paramref name="newV"/>.
    /// Does not change <see cref="CurrentLength"/> — call <see cref="AdvanceLength"/> after all layers.</summary>
    public void Append(IBackend backend, int layer, Tensor newK, Tensor newV)
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
            // First write: device-copy into an owned resident tensor via a single-input concat.
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

    /// <summary>Returns the layer's resident K prefix <c>[B, num_kv_heads, len, head_dim]</c>.</summary>
    public Tensor GetK(int layer)
    {
        ThrowIfDisposed();
        return _k[layer] ?? throw new InvalidOperationException($"Layer {layer} K not populated.");
    }

    /// <summary>Returns the layer's resident V prefix <c>[B, num_kv_heads, len, head_dim]</c>.</summary>
    public Tensor GetV(int layer)
    {
        ThrowIfDisposed();
        return _v[layer] ?? throw new InvalidOperationException($"Layer {layer} V not populated.");
    }

    /// <summary>Advances the shared position counter by <paramref name="by"/> tokens. Call once per
    /// transformer step, after every layer has appended.</summary>
    public void AdvanceLength(int by)
    {
        ThrowIfDisposed();
        if (by < 0) throw new ArgumentOutOfRangeException(nameof(by), by, "by must be >= 0");
        _currentLength += by;
    }

    /// <summary>Drops all stored K/V and resets the position counter.</summary>
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
        if (_disposed != 0) throw new ObjectDisposedException(nameof(DeviceKvCache));
    }

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
