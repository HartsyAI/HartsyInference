using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>One Qwen2 self-attention sublayer. GQA with biased Q/K/V projections
/// (Qwen2's defining feature vs Llama/Qwen3 — Llama has no Q/K/V bias). O projection has
/// no bias. RoPE is applied to Q and K after reshape into multi-head layout; GQA repeat
/// expands K/V to match Q-head count before SDPA.
///
/// <para>Supports both prefill (<c>T &gt; 1</c>, append all to cache) and incremental
/// decode (<c>T == 1</c>, append one position) via a shared
/// <see cref="StreamingKvCache"/>.</para></summary>
internal sealed unsafe class Qwen2Attention
{
    private readonly Qwen2Config _cfg;
    private readonly Qwen2RotaryEmbedding _rope;
    private readonly int _layerIndex;
    private readonly int _hidden;
    private readonly int _qDim;
    private readonly int _kvDim;
    private readonly int _headDim;

    private Tensor? _qW, _qB;
    private Tensor? _kW, _kB;
    private Tensor? _vW, _vB;
    private Tensor? _oW;     // no bias

    public Qwen2Attention(Qwen2Config cfg, Qwen2RotaryEmbedding rope, int layerIndex)
    {
        _cfg = cfg;
        _rope = rope;
        _layerIndex = layerIndex;
        _hidden = cfg.HiddenSize;
        _headDim = cfg.HeadDim;
        _qDim = cfg.NumAttentionHeads * cfg.HeadDim;     // = hidden when head_dim*num_heads = hidden
        _kvDim = cfg.KvHiddenSize;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.q_proj.weight"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.k_proj.weight"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.v_proj.weight"]);
        if (_cfg.AttentionBias)
        {
            _qB = WhisperOps.EnsureF32(w[$"{prefix}.q_proj.bias"]);
            _kB = WhisperOps.EnsureF32(w[$"{prefix}.k_proj.bias"]);
            _vB = WhisperOps.EnsureF32(w[$"{prefix}.v_proj.bias"]);
        }
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.o_proj.weight"]);
    }

    /// <summary>Self-attention forward. <paramref name="hidden"/> is <c>[B, T, hidden]</c>
    /// (channels-last). <paramref name="cache"/> contains keys + values from earlier
    /// positions; we append the new T positions and run SDPA against the full prefix.
    /// <paramref name="posStart"/> = <c>cache.CurrentLength</c> when called.
    ///
    /// <para>Returns a fresh <c>[B, T, hidden]</c> tensor with the attention output (post
    /// o_proj, before the residual add — caller owns the residual sum). The caller is
    /// also responsible for calling <see cref="StreamingKvCache.AdvanceLength"/> AFTER
    /// all layers have run (once per step).</para></summary>
    public Tensor Forward(IBackend backend, Tensor hidden, int batch, int t, int posStart, StreamingKvCache cache, Tensor? causalMask)
    {
        int hq = _cfg.NumAttentionHeads;
        int hkv = _cfg.NumKeyValueHeads;
        int d = _headDim;

        // 1. Q/K/V projections via ProjectLinear (handles bias broadcast).
        Tensor q = WhisperOps.ProjectLinear(backend, hidden, _qW!, _qB, batch, t, _hidden, _qDim);
        Tensor k = WhisperOps.ProjectLinear(backend, hidden, _kW!, _kB, batch, t, _hidden, _kvDim);
        Tensor v = WhisperOps.ProjectLinear(backend, hidden, _vW!, _vB, batch, t, _hidden, _kvDim);

        // 2. Reshape to multi-head: [B, T, H*D] → [B, H, T, D].
        Tensor qMh = new(new TensorShape(batch, hq, t, d), DType.F32);
        Tensor kMh = new(new TensorShape(batch, hkv, t, d), DType.F32);
        Tensor vMh = new(new TensorShape(batch, hkv, t, d), DType.F32);
        FlatToMultiHead(qMh, q, batch, t, hq, d); q.Dispose();
        FlatToMultiHead(kMh, k, batch, t, hkv, d); k.Dispose();
        FlatToMultiHead(vMh, v, batch, t, hkv, d); v.Dispose();

        // 3. RoPE on Q and K at absolute positions [posStart, posStart+t).
        _rope.EnsureBuilt(posStart + t);
        _rope.Apply(qMh, batch, hq, t, posStart);
        _rope.Apply(kMh, batch, hkv, t, posStart);

        // 4. Append new K, V to the cache at posStart..posStart+t-1.
        cache.Append(_layerIndex, kMh, vMh);
        kMh.Dispose();
        vMh.Dispose();

        // 5. Read back the full populated cache prefix [B, Hkv, posStart+t, D] for SDPA.
        int kvLen = posStart + t;
        Tensor cachedK = cache.GetK(_layerIndex);     // shape [B, Hkv, maxSeq, D] — but only first kvLen positions are valid.
        Tensor cachedV = cache.GetV(_layerIndex);
        Tensor kPrefix = SliceLeading(cachedK, batch, hkv, kvLen, d);
        Tensor vPrefix = SliceLeading(cachedV, batch, hkv, kvLen, d);

        // 6. GQA: repeat K/V groups so the head count matches Q.
        int groupSize = _cfg.KvGroupSize;
        Tensor kRep, vRep;
        if (groupSize == 1) { kRep = kPrefix; vRep = vPrefix; }
        else
        {
            kRep = new Tensor(new TensorShape(batch, hq, kvLen, d), DType.F32);
            vRep = new Tensor(new TensorShape(batch, hq, kvLen, d), DType.F32);
            RepeatKvHeads(kRep, kPrefix, batch, hkv, groupSize, kvLen, d);
            RepeatKvHeads(vRep, vPrefix, batch, hkv, groupSize, kvLen, d);
            kPrefix.Dispose();
            vPrefix.Dispose();
        }

        // 7. SDPA. SharpInference's ScaledDotProductAttention assumes
        // [B, H, S_q, D] for query, [B, H, S_k, D] for key/value, and a mask of shape
        // [S_q, S_k] (or broadcastable). For Qwen2 AR the natural mask is upper-triangular
        // when posStart=0 (prefill). For incremental decode (t=1, posStart>0) no mask is
        // needed since the single query attends to all keys.
        float scale = 1f / MathF.Sqrt(d);
        Tensor attnOut = new(new TensorShape(batch, hq, t, d), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kRep, vRep, causalMask, scale);
        qMh.Dispose();
        if (groupSize != 1) { kRep.Dispose(); vRep.Dispose(); }
        else { kPrefix.Dispose(); vPrefix.Dispose(); }

        // 8. Reshape multi-head → flat [B, T, hidden] then o_proj.
        Tensor attnFlat = new(new TensorShape(batch, t, _qDim), DType.F32);
        MultiHeadToFlat(attnFlat, attnOut, batch, t, hq, d);
        attnOut.Dispose();

        Tensor output = WhisperOps.ProjectLinear(backend, attnFlat, _oW!, bias: null, batch, t, _qDim, _hidden);
        attnFlat.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Copies the first <paramref name="kvLen"/> positions of a
    /// <c>[B, Hkv, maxSeq, D]</c> cache tensor into a packed
    /// <c>[B, Hkv, kvLen, D]</c> tensor suitable for SDPA. The cache itself is allocated
    /// once at maxSeq — slicing avoids passing the padded suffix to the kernel.</summary>
    private static Tensor SliceLeading(Tensor cacheTensor, int batch, int heads, int kvLen, int headDim)
    {
        int maxSeq = (int)cacheTensor.Shape[2];
        Tensor sliced = new(new TensorShape(batch, heads, kvLen, headDim), DType.F32);
        float* src = (float*)cacheTensor.DataPointer;
        float* dst = (float*)sliced.DataPointer;
        long perBatchHead_src = (long)maxSeq * headDim;
        long perBatchHead_dst = (long)kvLen * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                long srcOff = ((long)b * heads + h) * perBatchHead_src;
                long dstOff = ((long)b * heads + h) * perBatchHead_dst;
                Buffer.MemoryCopy(src + srcOff, dst + dstOff, perBatchHead_dst * 4, perBatchHead_dst * 4);
            }
        }
        return sliced;
    }

    // [B, S, H*D] → [B, H, S, D]
    private static void FlatToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int heads, int headDim)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                {
                    long inOff = ((long)b * seqLen + s) * heads * headDim + (long)h * headDim;
                    long outOff = (((long)b * heads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(ip + inOff, op + outOff, headDim * 4, headDim * 4);
                }
    }

    // [B, H, S, D] → [B, S, H*D]
    private static void MultiHeadToFlat(Tensor output, Tensor input, int batch, int seqLen, int heads, int headDim)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                {
                    long inOff = (((long)b * heads + h) * seqLen + s) * headDim;
                    long outOff = ((long)b * seqLen + s) * heads * headDim + (long)h * headDim;
                    Buffer.MemoryCopy(ip + inOff, op + outOff, headDim * 4, headDim * 4);
                }
    }

    // GQA: replicate each KV head `groupSize` times. [B, Hkv, S, D] → [B, Hkv*groupSize, S, D].
    private static void RepeatKvHeads(Tensor output, Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < kvHeads; h++)
            {
                long srcOff = ((long)b * kvHeads + h) * perHead;
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                    Buffer.MemoryCopy(ip + srcOff, op + dstOff, perHead * 4, perHead * 4);
                }
            }
        }
    }
}
