using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>One Dia attention sublayer, parameterized to cover all three flavors: encoder self-attention
/// (non-causal MHA), decoder self-attention (causal GQA, RoPE, KV-cached), and decoder cross-attention
/// (MHA over precomputed encoder K/V, no RoPE). No projection biases (Dia uses none). Reuses
/// <see cref="RotaryEmbedding"/>, <see cref="WhisperOps.ProjectLinear"/>, and
/// <see cref="IBackend.ScaledDotProductAttention"/>.</summary>
public sealed unsafe class DiaAttention
{
    private readonly int _qHeads, _kvHeads, _headDim, _qInDim, _kvInDim, _outDim;
    private readonly bool _useRope;
    private readonly bool _interleavedRope;
    private readonly float _attnScale;
    private Tensor? _qW, _kW, _vW, _oW;
    // Cross-attention precomputed K/V (constant across decode steps).
    private Tensor? _crossK, _crossV;
    private int _crossLen;

    /// <param name="attnScale">Softmax logit scale. Dia pre-scales Q and passes 1.0; models that rely on SDPA's
    /// default scaling (e.g. Zonos) pass <c>1/√head_dim</c>.</param>
    public DiaAttention(int qHeads, int kvHeads, int headDim, int qInDim, int kvInDim, int outDim, bool useRope,
        bool interleavedRope = false, float attnScale = 1f)
    {
        _qHeads = qHeads; _kvHeads = kvHeads; _headDim = headDim;
        _qInDim = qInDim; _kvInDim = kvInDim; _outDim = outDim; _useRope = useRope;
        _interleavedRope = interleavedRope; _attnScale = attnScale;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.q_proj.weight"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.k_proj.weight"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.v_proj.weight"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.o_proj.weight"]);
    }

    /// <summary>Self-attention. <paramref name="cache"/> + <paramref name="layerIndex"/> enable incremental
    /// decode (null = full-sequence encoder prefill). <paramref name="rope"/> tables are required when this
    /// attention was constructed with <c>useRope</c>.</summary>
    public Tensor SelfForward(IBackend backend, Tensor x, int t, int posStart, StreamingKvCache? cache,
        int layerIndex, Tensor? causalMask, float[]? cos, float[]? sin)
    {
        int qDim = _qHeads * _headDim, kvDim = _kvHeads * _headDim;
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, null, 1, t, _qInDim, qDim);
        Tensor k = WhisperOps.ProjectLinear(backend, x, _kW!, null, 1, t, _kvInDim, kvDim);
        Tensor v = WhisperOps.ProjectLinear(backend, x, _vW!, null, 1, t, _kvInDim, kvDim);

        Tensor qMh = new(new TensorShape(1, _qHeads, t, _headDim), DType.F32);
        Tensor kMh = new(new TensorShape(1, _kvHeads, t, _headDim), DType.F32);
        Tensor vMh = new(new TensorShape(1, _kvHeads, t, _headDim), DType.F32);
        DiaHeads.FlatToHeads(qMh, q, t, _qHeads, _headDim); q.Dispose();
        DiaHeads.FlatToHeads(kMh, k, t, _kvHeads, _headDim); k.Dispose();
        DiaHeads.FlatToHeads(vMh, v, t, _kvHeads, _headDim); v.Dispose();

        if (_useRope)
        {
            // Dia uses split-half (NeoX) RoPE (chunk(x, 2)); Zonos uses the GPT-J interleaved form.
            if (_interleavedRope)
            {
                DiaWeights.RopeInterleavedInPlace(qMh, _qHeads, t, _headDim, posStart, cos!, sin!);
                DiaWeights.RopeInterleavedInPlace(kMh, _kvHeads, t, _headDim, posStart, cos!, sin!);
            }
            else
            {
                DiaWeights.RopeSplitHalfInPlace(qMh, _qHeads, t, _headDim, posStart, cos!, sin!);
                DiaWeights.RopeSplitHalfInPlace(kMh, _kvHeads, t, _headDim, posStart, cos!, sin!);
            }
        }

        Tensor kUse, vUse;
        int kvLen;
        if (cache is not null)
        {
            cache.Append(layerIndex, kMh, vMh); kMh.Dispose(); vMh.Dispose();
            kvLen = posStart + t;
            kUse = DiaHeads.SliceLeading(cache.GetK(layerIndex), _kvHeads, kvLen, _headDim);
            vUse = DiaHeads.SliceLeading(cache.GetV(layerIndex), _kvHeads, kvLen, _headDim);
        }
        else { kUse = kMh; vUse = vMh; kvLen = t; }

        Tensor result = Attend(backend, qMh, kUse, vUse, t, kvLen, causalMask);
        qMh.Dispose();
        if (cache is not null || !ReferenceEquals(kUse, kMh)) { kUse.Dispose(); vUse.Dispose(); }
        return result;
    }

    /// <summary>GPU-resident self-attention (decode + prefill) for autoregressive decoders on a
    /// <see cref="IBackend.FlashDecodeSupported"/> backend. Mirrors the LLM GenericTransformer decode path: Q/K/V
    /// project straight into head-shaped tensors, GPU interleaved RoPE, in-place append into a fixed-capacity
    /// device KV cache, then GQA-aware FlashAttention over the resident buffer — no host reshape/rope/repeat glue
    /// and no O(n²) K/V re-upload. <paramref name="cosGpu"/>/<paramref name="sinGpu"/> are <c>[1, t, headDim]</c>
    /// (first headDim/2 lanes = the interleaved-pair angles for absolute positions [posStart, posStart+t)). The
    /// caller advances the cache length once after all layers. Requires the interleaved RoPE form (Zonos).</summary>
    public Tensor SelfForwardFlash(IBackend backend, Tensor x, int t, int posStart, IKvCache cache, int layerIndex,
        Tensor cosGpu, Tensor sinGpu)
    {
        // Project directly into head-shaped [1, t, H, headDim] (identical memory to [1, t, H*headDim]); avoids the
        // host FlatToHeads copy and keeps the activation GPU-resident.
        Tensor q = new(new TensorShape(1, t, _qHeads, _headDim), DType.F32);
        Tensor k = new(new TensorShape(1, t, _kvHeads, _headDim), DType.F32);
        Tensor v = new(new TensorShape(1, t, _kvHeads, _headDim), DType.F32);
        backend.Linear(q, x, _qW!, null);
        backend.Linear(k, x, _kW!, null);
        backend.Linear(v, x, _vW!, null);

        if (_useRope)
        {
            backend.ApplyRopeInterleaved(q, cosGpu, sinGpu);
            backend.ApplyRopeInterleaved(k, cosGpu, sinGpu);
        }

        // [1, t, H, D] → [1, H, t, D] for cache append + flash (GPU permute keeps residency).
        Tensor qMh = new(new TensorShape(1, _qHeads, t, _headDim), DType.F32);
        Tensor kMh = new(new TensorShape(1, _kvHeads, t, _headDim), DType.F32);
        Tensor vMh = new(new TensorShape(1, _kvHeads, t, _headDim), DType.F32);
        backend.Permute0213(qMh, q, t, _qHeads, _headDim); q.Dispose();
        backend.Permute0213(kMh, k, t, _kvHeads, _headDim); k.Dispose();
        backend.Permute0213(vMh, v, t, _kvHeads, _headDim); v.Dispose();

        cache.AppendStep(backend, layerIndex, kMh, vMh); kMh.Dispose(); vMh.Dispose();
        Tensor kFull = cache.KeyPrefix(layerIndex);   // [1, Hkv, maxSeq, D], valid = posStart + t
        Tensor vFull = cache.ValuePrefix(layerIndex);
        int kvLen = posStart + t, group = _qHeads / _kvHeads;

        Tensor attn = new(new TensorShape(1, _qHeads, t, _headDim), DType.F32);
        backend.FlashAttention(attn, qMh, kFull, vFull, kvLen, group, causal: true, qOffset: posStart, _attnScale);
        qMh.Dispose();

        Tensor flat = new(new TensorShape(1, t, _qHeads * _headDim), DType.F32);
        backend.Permute0213(flat, attn, _qHeads, t, _headDim); attn.Dispose();   // [1, H, t, D] → [1, t, H*D]
        Tensor outT = new(new TensorShape(1, t, _outDim), DType.F32);
        backend.Linear(outT, flat, _oW!, null); flat.Dispose();
        return outT;
    }

    /// <summary>Projects and caches the encoder K/V once for cross-attention.</summary>
    public void PrecomputeCrossKv(IBackend backend, Tensor encOut, int encLen)
    {
        int kvDim = _kvHeads * _headDim;
        Tensor k = WhisperOps.ProjectLinear(backend, encOut, _kW!, null, 1, encLen, _kvInDim, kvDim);
        Tensor v = WhisperOps.ProjectLinear(backend, encOut, _vW!, null, 1, encLen, _kvInDim, kvDim);
        _crossK = new Tensor(new TensorShape(1, _kvHeads, encLen, _headDim), DType.F32);
        _crossV = new Tensor(new TensorShape(1, _kvHeads, encLen, _headDim), DType.F32);
        DiaHeads.FlatToHeads(_crossK, k, encLen, _kvHeads, _headDim); k.Dispose();
        DiaHeads.FlatToHeads(_crossV, v, encLen, _kvHeads, _headDim); v.Dispose();
        _crossLen = encLen;
    }

    /// <summary>Cross-attention against the precomputed encoder K/V (call <see cref="PrecomputeCrossKv"/> first).</summary>
    public Tensor CrossForward(IBackend backend, Tensor x, int t)
    {
        int qDim = _qHeads * _headDim;
        Tensor q = WhisperOps.ProjectLinear(backend, x, _qW!, null, 1, t, _qInDim, qDim);
        Tensor qMh = new(new TensorShape(1, _qHeads, t, _headDim), DType.F32);
        DiaHeads.FlatToHeads(qMh, q, t, _qHeads, _headDim); q.Dispose();
        Tensor result = Attend(backend, qMh, _crossK!, _crossV!, t, _crossLen, null);
        qMh.Dispose();
        return result;
    }

    private Tensor Attend(IBackend backend, Tensor qMh, Tensor kMh, Tensor vMh, int t, int kvLen, Tensor? mask)
    {
        int group = _qHeads / _kvHeads;
        Tensor kRep, vRep;
        if (group == 1) { kRep = kMh; vRep = vMh; }
        else
        {
            kRep = new Tensor(new TensorShape(1, _qHeads, kvLen, _headDim), DType.F32);
            vRep = new Tensor(new TensorShape(1, _qHeads, kvLen, _headDim), DType.F32);
            DiaHeads.RepeatKv(kRep, kMh, _kvHeads, group, kvLen, _headDim);
            DiaHeads.RepeatKv(vRep, vMh, _kvHeads, group, kvLen, _headDim);
        }
        // Dia pre-scales Q so its scale is 1.0; Zonos relies on SDPA's default 1/sqrt(head_dim) (see ctor).
        Tensor attn = new(new TensorShape(1, _qHeads, t, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kRep, vRep, mask, _attnScale);
        if (group != 1) { kRep.Dispose(); vRep.Dispose(); }
        Tensor flat = new(new TensorShape(1, t, _qHeads * _headDim), DType.F32);
        DiaHeads.HeadsToFlat(flat, attn, t, _qHeads, _headDim); attn.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, flat, _oW!, null, 1, t, _qHeads * _headDim, _outDim);
        flat.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qW, _kW, _vW, _oW];
        foreach (Tensor? x in all) if (x is not null) yield return x;
    }
}
