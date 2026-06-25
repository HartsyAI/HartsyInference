using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Chatterbox;

/// <summary>Chatterbox T3 cond-prompt Perceiver resampler (<c>cond_enc.perceiver</c>): a fixed set of 32
/// learned query tokens cross-attend the 150 prompt-speech embeddings, then self-attend, producing 32
/// conditioning tokens. One shared <see cref="AttentionBlock"/> is applied twice (cross then self). Standard
/// multi-head attention (4 heads, head_dim 256, scale 256^-0.5); the upstream einsum fallback is unused since
/// the model runs the flash/SDPA path.</summary>
public sealed unsafe class ChatterboxPerceiver : IDisposable
{
    private const int Dim = 1024;
    private const int Heads = 4;
    private const int HeadDim = Dim / Heads;     // 256
    private const int NumQueries = 32;

    private Tensor? _query;          // pre_attention_query [1, 32, 1024]
    private Tensor? _normW, _normB, _qW, _qB, _kW, _kB, _vW, _vB, _outW, _outB;
    private int _disposed;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "cond_enc.perceiver")
    {
        _query = WhisperOps.EnsureF32(w[$"{prefix}.pre_attention_query"]);
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.attn.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.attn.norm.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_q.weight"]); _qB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_q.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_k.weight"]); _kB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_k.bias"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_v.weight"]); _vB = WhisperOps.EnsureF32(w[$"{prefix}.attn.to_v.bias"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.attn.proj_out.weight"]); _outB = WhisperOps.EnsureF32(w[$"{prefix}.attn.proj_out.bias"]);
    }

    // TODO(gpu-residency): ToHeads/FromHeads and the residual add use host `(float*)DataPointer` loops, forcing
    // device→host syncs on CUDA. Runs once per synthesis (cheap), but port to backend ops/PTX for full residency.
    /// <summary>Resamples prompt-speech embeddings <c>[1, T, 1024]</c> → conditioning <c>[1, 32, 1024]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor promptEmb)
    {
        if (_query is null) throw new InvalidOperationException("ChatterboxPerceiver weights not loaded.");
        // query is the learned [1, 32, 1024]; clone it so the residual add owns its buffer.
        Tensor query = new(new TensorShape(1, NumQueries, Dim), DType.F32);
        backend.CopyTo(query, _query!);

        Tensor preAtt = AttentionBlock(backend, query, promptEmb);   // cross-attention (queries → prompt)
        query.Dispose();
        Tensor outT = AttentionBlock(backend, preAtt, preAtt);       // self-attention
        preAtt.Dispose();
        return outT;
    }

    /// <summary>AttentionBlock2: x1n=LN(x1), x2n=LN(x2) (shared norm); q=Wq·x1n, k=Wk·x2n, v=Wv·x2n; MHA;
    /// h=Wout·attn; return x1 + h.</summary>
    private Tensor AttentionBlock(IBackend backend, Tensor x1, Tensor x2)
    {
        int sq = (int)x1.Shape[1], sk = (int)x2.Shape[1];
        Tensor x1n = new(x1.Shape, DType.F32); backend.LayerNorm(x1n, x1, _normW!, _normB!, 1e-5f);
        Tensor x2n = ReferenceEquals(x1, x2) ? x1n : NewLayerNorm(backend, x2);

        Tensor q = WhisperOps.ProjectLinear(backend, x1n, _qW!, _qB, 1, sq, Dim, Dim);
        Tensor k = WhisperOps.ProjectLinear(backend, x2n, _kW!, _kB, 1, sk, Dim, Dim);
        Tensor v = WhisperOps.ProjectLinear(backend, x2n, _vW!, _vB, 1, sk, Dim, Dim);
        x1n.Dispose();
        if (!ReferenceEquals(x2n, x1n)) x2n.Dispose();

        Tensor qH = ToHeads(q, sq), kH = ToHeads(k, sk), vH = ToHeads(v, sk);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, Heads, sq, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, mask: null, 1f / MathF.Sqrt(HeadDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();
        Tensor merged = FromHeads(attn, sq);
        attn.Dispose();
        Tensor proj = WhisperOps.ProjectLinear(backend, merged, _outW!, _outB, 1, sq, Dim, Dim);
        merged.Dispose();

        // Residual with x1 (NOT the normed x1).
        float* pp = (float*)proj.DataPointer;
        float* x1p = (float*)x1.DataPointer;
        long n = proj.ElementCount;
        for (long i = 0; i < n; i++) pp[i] += x1p[i];
        return proj;
    }

    private Tensor NewLayerNorm(IBackend backend, Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.LayerNorm(o, x, _normW!, _normB!, 1e-5f);
        return o;
    }

    private static Tensor ToHeads(Tensor seq, int t)
    {
        Tensor outT = new(new TensorShape(1, Heads, t, HeadDim), DType.F32);
        float* ip = (float*)seq.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < Heads; h++)
            for (int j = 0; j < t; j++)
                for (int d = 0; d < HeadDim; d++)
                    op[((long)h * t + j) * HeadDim + d] = ip[(long)j * Dim + h * HeadDim + d];
        return outT;
    }

    private static Tensor FromHeads(Tensor heads, int t)
    {
        Tensor outT = new(new TensorShape(1, t, Dim), DType.F32);
        float* ip = (float*)heads.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int h = 0; h < Heads; h++)
            for (int j = 0; j < t; j++)
                for (int d = 0; d < HeadDim; d++)
                    op[(long)j * Dim + h * HeadDim + d] = ip[((long)h * t + j) * HeadDim + d];
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_query, _normW, _normB, _qW, _qB, _kW, _kB, _vW, _vB, _outW, _outB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
