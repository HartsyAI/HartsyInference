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
    private int _disposed;

    public int NumLayers => _k.Length;
    public int BatchSize { get; }
    public int NumKvHeads { get; }
    public int HeadDim { get; }
    public int MaxSequenceLength { get; }

    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Allocates per-layer fixed buffers sized for <paramref name="maxSequenceLength"/> tokens.</summary>
    public FixedKvCache(int numLayers, int batch, int numKvHeads, int headDim, int maxSequenceLength)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (batch != 1) throw new NotSupportedException("FixedKvCache supports batch=1.");
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));

        BatchSize = batch;
        NumKvHeads = numKvHeads;
        HeadDim = headDim;
        MaxSequenceLength = maxSequenceLength;
        _k = new Tensor[numLayers];
        _v = new Tensor[numLayers];
        TensorShape shape = new(1, numKvHeads, maxSequenceLength, headDim);
        for (int i = 0; i < numLayers; i++)
        {
            _k[i] = new Tensor(shape, DType.F32);
            _v[i] = new Tensor(shape, DType.F32);
        }
    }

    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
    {
        ThrowIfDisposed();
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
