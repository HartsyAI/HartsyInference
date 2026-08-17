using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Fixed-capacity device KV cache: each layer holds a single pre-allocated <c>[1, num_kv_heads,
/// maxSeq, head_dim]</c> buffer that new K/V are written into <b>in place</b> (<see cref="IBackend.KvCacheAppend"/>).
/// Unlike the <see cref="KvCache"/> (which grows via <c>Concat</c>, reallocating + copying the whole prefix
/// every token — O(n²) total), appends here are O(tNew) and VRAM is bounded by maxSeq up front. The buffer's
/// sequence stride (maxSeq) exceeds the valid length, so FlashAttention is given the valid key count separately
/// and reads the stride from the tensor shape.
///
/// <para>This is the single-sequence step toward block-paged KV + continuous batching: it removes the O(n²)
/// growth and fixes the footprint. Multi-sequence block paging builds on this.</para>
///
/// <para>The full cap is allocated up front on purpose. Growing the buffer in chunks instead was built and
/// measured on 2026-08-17: it cost ~4 GB of peak VRAM and ~20% of the decode stage, because the device mempool
/// runs at a keep-everything release threshold and so retains every intermediate buffer size. Use
/// <see cref="PagedKvCache"/> if a bounded footprint is needed — uniform page sizes are what that pool rewards.
/// </para></summary>
public sealed class FixedKvCache : IKvCache, IDisposable
{
    private readonly Tensor?[] _k;
    private readonly Tensor?[] _v;
    private readonly int[] _headDimPerLayer;
    private readonly int[] _capacity;
    private readonly DType _dtype;
    private int _currentLength;
    /// <summary>Per-layer device residency: each layer's K/V allocate on the backend that first APPENDS to that
    /// layer. Under multi-device layer-split placement this puts every layer's KV on its stage's device with no
    /// placement API at all; it also stops shared-KV-slot layers (Gemma-4) from ever going device-resident, since
    /// nothing appends to them.</summary>
    private readonly bool[] _residentLayer;
    private int _disposed;

    public int NumLayers => _k.Length;
    public int BatchSize { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }
    /// <summary>The logical cap an append may never cross; with grow-on-demand it is NOT what is allocated.</summary>
    public int MaxSequenceLength { get; }

    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }


    /// <summary>Tokens layer <paramref name="layer"/> currently has room for; equals
    /// <see cref="MaxSequenceLength"/> unless the cache grows on demand.</summary>
    public int LayerCapacity(int layer) { ThrowIfDisposed(); return _capacity[layer]; }

    /// <summary>Allocates per-layer fixed buffers sized for <paramref name="maxSequenceLength"/> tokens, all
    /// layers sharing one <paramref name="headDim"/> (every architecture except Gemma-4).</summary>
    public FixedKvCache(int numLayers, int batch, int numKvHeads, int headDim, int maxSequenceLength,
        DType? kvDtype = null)
        : this(numLayers, batch, numKvHeads, UniformHeadDims(numLayers, headDim), maxSequenceLength, kvDtype) { }

    /// <summary>Allocates per-layer fixed buffers with a PER-LAYER head dimension (Gemma-4: local/SWA layers are
    /// narrower than global layers). <paramref name="headDimPerLayer"/> must have <paramref name="numLayers"/>
    /// entries — a layer that shares another layer's KV cache slot (see <see cref="TransformerConfig.HasOwnKv"/>)
    /// still gets an entry here (simplest to allocate and just never write/read it) sized to its OWN head dim,
    /// even though nothing ever appends to it.</summary>
    /// <param name="kvDtype">Storage dtype for the K/V buffers. Default F32 (unchanged behavior). F16 halves
    /// the cache's VRAM footprint (K/V straight out of the projection stay F32 — <see cref="IBackend.KvCacheAppend"/>
    /// converts on write; <see cref="IBackend.FlashAttention"/> upconverts back to F32 on read, so compute is
    /// unaffected — only storage is narrower). Opt-in: the CUDA kernels support it (v1: monolithic FlashAttention
    /// only, not the split-K or graph-decode fast paths, which fall back to the monolithic kernel automatically),
    /// but this isn't the default until it's soaked — same reasoning as <c>DeviceGate</c>'s concurrent-mode flag.</param>
    public FixedKvCache(int numLayers, int batch, int numKvHeads, int[] headDimPerLayer, int maxSequenceLength,
        DType? kvDtype = null)
    {
        DType dtype = kvDtype ?? DType.F32;
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (batch != 1) throw new NotSupportedException("FixedKvCache supports batch=1.");
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDimPerLayer.Length != numLayers) throw new ArgumentException($"headDimPerLayer has {headDimPerLayer.Length} entries, expected {numLayers}.", nameof(headDimPerLayer));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (dtype != DType.F32 && dtype != DType.F16)
            throw new ArgumentOutOfRangeException(nameof(kvDtype), $"FixedKvCache supports F32 or F16 storage; got {dtype}.");

        BatchSize = batch;
        NumKvHeads = numKvHeads;
        HeadDim = headDimPerLayer[0];   // best-effort single-value summary; per-layer callers use KeyPrefix/ValuePrefix shapes directly
        MaxSequenceLength = maxSequenceLength;
        _dtype = dtype;
        _headDimPerLayer = headDimPerLayer;
        _capacity = new int[numLayers];
        _k = new Tensor?[numLayers];
        _v = new Tensor?[numLayers];
        _residentLayer = new bool[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            int hd = headDimPerLayer[i];
            if (hd <= 0) throw new ArgumentOutOfRangeException(nameof(headDimPerLayer), $"layer {i} head dim must be positive.");
            _capacity[i] = maxSequenceLength;
            TensorShape shape = new(1, numKvHeads, maxSequenceLength, hd);
            _k[i] = new Tensor(shape, dtype);
            _v[i] = new Tensor(shape, dtype);
        }
    }

    private static int[] UniformHeadDims(int numLayers, int headDim)
    {
        int[] a = new int[numLayers];
        Array.Fill(a, headDim);
        return a;
    }

    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
    {
        ThrowIfDisposed();
        int tNew = (int)newK.Shape[2];
        if (_currentLength + tNew > MaxSequenceLength)
            throw new InvalidOperationException($"FixedKvCache overflow: current={_currentLength}, adding={tNew}, max={MaxSequenceLength}.");
        // First append TO THIS LAYER: place its K/V buffers directly on the appending backend's device (no
        // per-buffer H2D of the zeroed host allocation). Per-layer rather than all-at-once so a layer-split
        // placement lands each layer's KV on its own stage's device, and never-appended layers (shared KV slots)
        // never go resident. Backends without a device treat this as a no-op and fall back to lazy upload. The
        // tail beyond the valid length is never read, so the buffers are left uninitialized.
        if (!_residentLayer[layer])
        {
            _residentLayer[layer] = true;
            backend.ResidentAllocateKv(_k[layer]!);
            backend.ResidentAllocateKv(_v[layer]!);
        }
        backend.KvCacheAppend(_k[layer]!, newK, _currentLength);
        backend.KvCacheAppend(_v[layer]!, newV, _currentLength);
    }


    /// <summary>The layer's K buffer <c>[1, num_kv_heads, capacity, head_dim]</c>; valid keys are the first
    /// <see cref="CurrentLength"/> (+ the just-appended step). FlashAttention is told the valid length.</summary>
    public Tensor KeyPrefix(int layer) { ThrowIfDisposed(); return Buffer(_k, layer); }

    public Tensor ValuePrefix(int layer) { ThrowIfDisposed(); return Buffer(_v, layer); }

    private static Tensor Buffer(Tensor?[] buffers, int layer) => buffers[layer]
        ?? throw new InvalidOperationException($"FixedKvCache layer {layer} has no buffer yet — grow-on-demand allocates it on the first AppendStep.");

    public void AdvanceLength(int by)
    {
        ThrowIfDisposed();
        if (by < 0) throw new ArgumentOutOfRangeException(nameof(by));
        _currentLength += by;
    }

    /// <summary>Rolls back to <paramref name="newLength"/> (speculative-decode rejection). No physical
    /// erasure needed: the buffer is a single fixed array written in place, and every read already scopes
    /// itself to <see cref="CurrentLength"/> via the caller-supplied valid-length, so shrinking the counter
    /// is sufficient — a subsequent <see cref="AppendStep"/> simply overwrites whatever was beyond it.</summary>
    public void Truncate(int newLength)
    {
        ThrowIfDisposed();
        if (newLength < 0 || newLength > _currentLength) throw new ArgumentOutOfRangeException(nameof(newLength));
        _currentLength = newLength;
    }

    public void Reset() { ThrowIfDisposed(); _currentLength = 0; }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(FixedKvCache));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor? t in _k) t?.Dispose();
        foreach (Tensor? t in _v) t?.Dispose();
    }
}
