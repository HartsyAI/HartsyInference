using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Streaming;

/// <summary>Pre-allocated per-layer K/V cache for autoregressive streaming inference.
/// Replaces the "concat at every step" pattern with a fixed-size buffer that the
/// generation loop writes into at the current position, then hands the full buffer to
/// the backend's attention op together with <see cref="CurrentLength"/>.
///
/// <para>Layout: <c>K[layer], V[layer]</c> each have shape
/// <c>[batch, num_kv_heads, max_seq_len, head_dim]</c>. <c>num_kv_heads</c> handles GQA
/// (e.g. VibeVoice Qwen2.5-1.5B has 12 query heads but 2 K/V heads). All layers share
/// one <see cref="CurrentLength"/> — every transformer step advances K/V at every layer
/// by the same number of new tokens.</para>
///
/// <para>This cache is intentionally allocation-free on the hot path: <see cref="Append"/>
/// is a pure memcpy into pre-allocated buffers. Re-use across utterances via
/// <see cref="Reset"/> (cheap — no buffer realloc, just reset position).</para>
///
/// <para>Used by streaming STT (Whisper-streaming, Parakeet streaming) and streaming TTS
/// AR loops (Bark, XTTS streaming, CosyVoice streaming, Sesame CSM, VibeVoice). Diffusion
/// pipelines don't need this — they consume the LM cache through the dotLLM dependency
/// directly. This cache is for in-house transformer paths that bypass dotLLM (e.g. the
/// non-LM decoders inside the audio models themselves).</para></summary>
public sealed class StreamingKvCache : IKvCache, IDisposable
{
    private readonly Tensor[] _k;
    private readonly Tensor[] _v;
    // Per-layer populated-prefix tensors for the IKvCache contract (cache-owned, refreshed on AppendStep).
    private readonly Tensor?[] _prefixK;
    private readonly Tensor?[] _prefixV;
    private int _currentLength;
    private int _disposed;

    public int NumLayers => _k.Length;
    public int BatchSize { get; }
    public int NumKvHeads { get; }
    public int MaxSequenceLength { get; }
    public int HeadDim { get; }

    /// <summary>Tokens currently stored in the cache. <c>0</c> immediately after
    /// construction and after <see cref="Reset"/>.</summary>
    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Free positions remaining before the cache is full.</summary>
    public int FreeSlots { get { ThrowIfDisposed(); return MaxSequenceLength - _currentLength; } }

    public StreamingKvCache(int numLayers, int batch, int numKvHeads, int maxSequenceLength, int headDim, DType? dtype = null)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be > 0");
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch must be > 0");
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads), numKvHeads, "numKvHeads must be > 0");
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), maxSequenceLength, "maxSequenceLength must be > 0");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "headDim must be > 0");

        BatchSize = batch;
        NumKvHeads = numKvHeads;
        MaxSequenceLength = maxSequenceLength;
        HeadDim = headDim;
        DType storageDtype = dtype ?? DType.F32;

        _k = new Tensor[numLayers];
        _v = new Tensor[numLayers];
        _prefixK = new Tensor?[numLayers];
        _prefixV = new Tensor?[numLayers];
        TensorShape shape = new(batch, numKvHeads, maxSequenceLength, headDim);
        for (int i = 0; i < numLayers; i++)
        {
            _k[i] = new Tensor(shape, storageDtype);
            _v[i] = new Tensor(shape, storageDtype);
        }
    }

    /// <summary>Returns the layer's full pre-allocated K buffer. Caller uses
    /// <see cref="CurrentLength"/> to know how many positions along the sequence axis
    /// are populated.</summary>
    public Tensor GetK(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_k.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), layerIndex, $"layerIndex must be in [0, {_k.Length}).");
        return _k[layerIndex];
    }

    /// <summary>Returns the layer's full pre-allocated V buffer. See <see cref="GetK"/>.</summary>
    public Tensor GetV(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_v.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), layerIndex, $"layerIndex must be in [0, {_v.Length}).");
        return _v[layerIndex];
    }

    /// <summary>Writes <paramref name="newK"/> and <paramref name="newV"/> for one layer
    /// at the current sequence offset. Both tensors must have shape
    /// <c>[batch, num_kv_heads, T_new, head_dim]</c>; <c>T_new</c> may be 1 (a single
    /// decode step) or larger (a prefill batch).
    ///
    /// <para>This call does NOT advance <see cref="CurrentLength"/> — that happens once
    /// per step via <see cref="AdvanceLength"/> after every layer has been appended to.
    /// Decouples the per-layer writes from the per-step position bump.</para>
    ///
    /// <para>Throws if <c>CurrentLength + T_new &gt; MaxSequenceLength</c>. Throws on
    /// dtype mismatch.</para></summary>
    public void Append(int layerIndex, Tensor newK, Tensor newV)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_k.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), layerIndex, $"layerIndex must be in [0, {_k.Length}).");
        ValidateNewTensor(newK, nameof(newK));
        ValidateNewTensor(newV, nameof(newV));

        int tNew = (int)newK.Shape[2];
        if (_currentLength + tNew > MaxSequenceLength)
            throw new InvalidOperationException($"StreamingKvCache overflow: current={_currentLength}, adding={tNew}, max={MaxSequenceLength}.");

        WriteAtOffset(_k[layerIndex], newK, _currentLength, tNew);
        WriteAtOffset(_v[layerIndex], newV, _currentLength, tNew);
    }

    /// <summary><see cref="IKvCache"/> append: writes the new K/V into the pre-allocated buffer (host memcpy)
    /// and refreshes the layer's cache-owned populated prefix used by <see cref="KeyPrefix"/>/<see cref="ValuePrefix"/>.
    /// <paramref name="backend"/> is unused (host cache). Does not advance the position counter.</summary>
    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
    {
        Append(layer, newK, newV);
        int len = _currentLength + (int)newK.Shape[2];
        _prefixK[layer]?.Dispose();
        _prefixV[layer]?.Dispose();
        _prefixK[layer] = SlicePrefix(_k[layer], len);
        _prefixV[layer] = SlicePrefix(_v[layer], len);
    }

    /// <summary><see cref="IKvCache"/> populated K prefix <c>[B, num_kv_heads, len, head_dim]</c> (cache-owned).</summary>
    public Tensor KeyPrefix(int layer)
    {
        ThrowIfDisposed();
        return _prefixK[layer] ?? throw new InvalidOperationException($"Layer {layer} prefix not populated.");
    }

    /// <summary><see cref="IKvCache"/> populated V prefix <c>[B, num_kv_heads, len, head_dim]</c> (cache-owned).</summary>
    public Tensor ValuePrefix(int layer)
    {
        ThrowIfDisposed();
        return _prefixV[layer] ?? throw new InvalidOperationException($"Layer {layer} prefix not populated.");
    }

    /// <summary>Copies the first <paramref name="len"/> sequence positions of a padded <c>[B, H, T_max, D]</c>
    /// buffer into a packed <c>[B, H, len, D]</c> tensor (per <c>(b, h)</c> contiguous memcpy).</summary>
    private unsafe Tensor SlicePrefix(Tensor padded, int len)
    {
        Tensor outT = new(new TensorShape(BatchSize, NumKvHeads, len, HeadDim), padded.DType);
        int elemBytes = padded.DType.SizeInBytes;
        long perCopy = (long)len * HeadDim * elemBytes;
        long srcStride = (long)MaxSequenceLength * HeadDim * elemBytes;
        long dstStride = (long)len * HeadDim * elemBytes;
        byte* sp = (byte*)padded.DataPointer;
        byte* dp = (byte*)outT.DataPointer;
        for (int bh = 0; bh < BatchSize * NumKvHeads; bh++)
            Buffer.MemoryCopy(sp + bh * srcStride, dp + bh * dstStride, perCopy, perCopy);
        return outT;
    }

    /// <summary>Advances the shared sequence position. Call once after every layer has
    /// been written for the current step.</summary>
    public void AdvanceLength(int by)
    {
        ThrowIfDisposed();
        if (by < 0) throw new ArgumentOutOfRangeException(nameof(by), by, "by must be >= 0");
        if (_currentLength + by > MaxSequenceLength)
            throw new InvalidOperationException($"StreamingKvCache advance overflow: current={_currentLength}, by={by}, max={MaxSequenceLength}.");
        _currentLength += by;
    }

    /// <summary>Resets the sequence position to 0 without touching the underlying
    /// buffers. The stale K/V data is left in place — it's never read past
    /// <see cref="CurrentLength"/> so leaving it is harmless and saves a memset.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        _currentLength = 0;
        for (int i = 0; i < _prefixK.Length; i++)
        {
            _prefixK[i]?.Dispose(); _prefixK[i] = null;
            _prefixV[i]?.Dispose(); _prefixV[i] = null;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _k) t.Dispose();
        foreach (Tensor t in _v) t.Dispose();
        foreach (Tensor? t in _prefixK) t?.Dispose();
        foreach (Tensor? t in _prefixV) t?.Dispose();
    }

    private void ValidateNewTensor(Tensor newKv, string paramName)
    {
        if (newKv.Shape.Rank != 4)
            throw new ArgumentException($"K/V tensor must be rank-4, got {newKv.Shape}.", paramName);
        if ((int)newKv.Shape[0] != BatchSize)
            throw new ArgumentException($"Batch mismatch: cache expects {BatchSize}, got {newKv.Shape[0]}.", paramName);
        if ((int)newKv.Shape[1] != NumKvHeads)
            throw new ArgumentException($"num_kv_heads mismatch: cache expects {NumKvHeads}, got {newKv.Shape[1]}.", paramName);
        if ((int)newKv.Shape[3] != HeadDim)
            throw new ArgumentException($"head_dim mismatch: cache expects {HeadDim}, got {newKv.Shape[3]}.", paramName);
        if (newKv.DType != _k[0].DType)
            throw new ArgumentException($"Dtype mismatch: cache stores {_k[0].DType}, got {newKv.DType}.", paramName);
    }

    /// <summary>Copies the rank-4 <paramref name="src"/> <c>[B, H, T_new, D]</c> into the
    /// rank-4 <paramref name="dst"/> <c>[B, H, T_max, D]</c> at sequence offset
    /// <paramref name="seqOffset"/>. Each <c>(b, h)</c> pair becomes a contiguous
    /// <c>T_new * D</c> memcpy thanks to D being the trailing dim.</summary>
    private void WriteAtOffset(Tensor dst, Tensor src, int seqOffset, int tNew)
    {
        int b = BatchSize;
        int h = NumKvHeads;
        int d = HeadDim;
        int tMax = MaxSequenceLength;
        int elemBytes = dst.DType.SizeInBytes;
        long perCopyBytes = (long)tNew * d * elemBytes;

        unsafe
        {
            byte* sp = (byte*)src.DataPointer;
            byte* dp = (byte*)dst.DataPointer;
            for (int bi = 0; bi < b; bi++)
            {
                for (int hi = 0; hi < h; hi++)
                {
                    long dstOffset = (((long)bi * h + hi) * tMax + seqOffset) * d * elemBytes;
                    long srcOffset = ((long)bi * h + hi) * tNew * d * elemBytes;
                    Buffer.MemoryCopy(sp + srcOffset, dp + dstOffset, perCopyBytes, perCopyBytes);
                }
            }
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }
}
