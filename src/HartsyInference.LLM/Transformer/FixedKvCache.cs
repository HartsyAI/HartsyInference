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
/// growth and fixes the footprint. Multi-sequence block paging builds on this.</para></summary>
public sealed class FixedKvCache : IKvCache, IDisposable
{
    private readonly Tensor[] _k;
    private readonly Tensor[] _v;
    private int _currentLength;
    private bool _resident;
    private int _disposed;

    public int NumLayers => _k.Length;
    public int BatchSize { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }
    public int MaxSequenceLength { get; }

    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Allocates per-layer fixed buffers sized for <paramref name="maxSequenceLength"/> tokens, all
    /// layers sharing one <paramref name="headDim"/> (every architecture except Gemma-4).</summary>
    public FixedKvCache(int numLayers, int batch, int numKvHeads, int headDim, int maxSequenceLength)
        : this(numLayers, batch, numKvHeads, UniformHeadDims(numLayers, headDim), maxSequenceLength) { }

    /// <summary>Allocates per-layer fixed buffers with a PER-LAYER head dimension (Gemma-4: local/SWA layers are
    /// narrower than global layers). <paramref name="headDimPerLayer"/> must have <paramref name="numLayers"/>
    /// entries — a layer that shares another layer's KV cache slot (see <see cref="TransformerConfig.HasOwnKv"/>)
    /// still gets an entry here (simplest to allocate and just never write/read it) sized to its OWN head dim,
    /// even though nothing ever appends to it.</summary>
    public FixedKvCache(int numLayers, int batch, int numKvHeads, int[] headDimPerLayer, int maxSequenceLength)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (batch != 1) throw new NotSupportedException("FixedKvCache supports batch=1.");
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDimPerLayer.Length != numLayers) throw new ArgumentException($"headDimPerLayer has {headDimPerLayer.Length} entries, expected {numLayers}.", nameof(headDimPerLayer));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));

        BatchSize = batch;
        NumKvHeads = numKvHeads;
        HeadDim = headDimPerLayer[0];   // best-effort single-value summary; per-layer callers use KeyPrefix/ValuePrefix shapes directly
        MaxSequenceLength = maxSequenceLength;
        _k = new Tensor[numLayers];
        _v = new Tensor[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            int hd = headDimPerLayer[i];
            if (hd <= 0) throw new ArgumentOutOfRangeException(nameof(headDimPerLayer), $"layer {i} head dim must be positive.");
            TensorShape shape = new(1, numKvHeads, maxSequenceLength, hd);
            _k[i] = new Tensor(shape, DType.F32);
            _v[i] = new Tensor(shape, DType.F32);
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
        // First append: place every layer's K/V buffer directly on the device (no per-buffer H2D of the zeroed
        // host allocation). Backends without a device treat this as a no-op and fall back to lazy upload. The tail
        // beyond the valid length is never read, so the buffers are left uninitialized.
        if (!_resident)
        {
            _resident = true;
            for (int i = 0; i < _k.Length; i++) { backend.ResidentAllocateKv(_k[i]); backend.ResidentAllocateKv(_v[i]); }
        }
        int tNew = (int)newK.Shape[2];
        if (_currentLength + tNew > MaxSequenceLength)
            throw new InvalidOperationException($"FixedKvCache overflow: current={_currentLength}, adding={tNew}, max={MaxSequenceLength}.");
        backend.KvCacheAppend(_k[layer], newK, _currentLength);
        backend.KvCacheAppend(_v[layer], newV, _currentLength);
    }

    /// <summary>The layer's K buffer <c>[1, num_kv_heads, maxSeq, head_dim]</c>; valid keys are the first
    /// <see cref="CurrentLength"/> (+ the just-appended step). FlashAttention is told the valid length.</summary>
    public Tensor KeyPrefix(int layer) { ThrowIfDisposed(); return _k[layer]; }

    public Tensor ValuePrefix(int layer) { ThrowIfDisposed(); return _v[layer]; }

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
        foreach (Tensor t in _k) t.Dispose();
        foreach (Tensor t in _v) t.Dispose();
    }
}
