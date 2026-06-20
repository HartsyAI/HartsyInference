using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Zonos transformer backbone — 26 pre-LayerNorm GQA blocks (interleaved RoPE, no bias) + fused
/// gate-up SwiGLU. Reuses <see cref="DiaAttention"/> (self-attention, GQA, interleaved RoPE, KV cache) and
/// <see cref="DiaMlp"/> (fused SwiGLU) by splitting Zonos's fused <c>in_proj</c> and remapping <c>fc1/fc2</c>
/// keys at load. The only Zonos-specific bit is LayerNorm (vs Llama's RMSNorm).</summary>
public sealed unsafe class ZonosBackbone : IDisposable
{
    private readonly ZonosConfig _cfg;
    private readonly ZonosBlock[] _blocks;
    private Tensor? _finalNormW, _finalNormB;
    private float[]? _cos, _sin;
    private int _disposed;

    public ZonosBackbone(ZonosConfig cfg)
    {
        _cfg = cfg;
        _blocks = new ZonosBlock[cfg.NumLayers];
        for (int i = 0; i < _blocks.Length; i++) _blocks[i] = new ZonosBlock(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "backbone")
    {
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{prefix}.layers.{i}");
        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.norm_f.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.norm_f.bias"]);
        (_cos, _sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, _cfg.MaxPositions);
    }

    /// <summary>Forward over prebuilt input embeddings <c>[1, T, hidden]</c> (prefix conditioning + audio
    /// embeds for prefill, or a single frame for decode).</summary>
    public Tensor Forward(IBackend backend, Tensor embeds, int t, int posStart, StreamingKvCache cache)
    {
        Tensor? mask = t > 1 ? BuildCausalMask(t, posStart) : null;
        try
        {
            Tensor hidden = embeds;
            bool owns = false;
            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].Forward(backend, hidden, t, posStart, cache, i, mask, _cos!, _sin!);
                if (owns) hidden.Dispose();
                hidden = next; owns = true;
            }
            cache.AdvanceLength(t);
            Tensor normed = new(hidden.Shape, DType.F32);
            backend.LayerNorm(normed, hidden, _finalNormW!, _finalNormB!, _cfg.NormEps);
            if (owns) hidden.Dispose();
            return normed;
        }
        finally { mask?.Dispose(); }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (ZonosBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_finalNormW is not null) yield return _finalNormW;
        if (_finalNormB is not null) yield return _finalNormB;
    }

    private static Tensor BuildCausalMask(int tq, int qOffset)
    {
        int tk = qOffset + tq;
        Tensor mask = new(new TensorShape(1, 1, tq, tk), DType.F32);
        float* p = (float*)mask.DataPointer;
        const float NegInf = -1e30f;
        for (int q = 0; q < tq; q++)
        {
            int absQ = qOffset + q;
            for (int k = 0; k < tk; k++) p[q * tk + k] = k <= absQ ? 0f : NegInf;
        }
        return mask;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _finalNormW = null; _finalNormB = null;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One Zonos block: <c>x += attn(LN(x)); x += mlp(LN2(x))</c>.</summary>
public sealed unsafe class ZonosBlock
{
    private readonly ZonosConfig _cfg;
    private readonly DiaAttention _attn;
    private readonly DiaMlp _mlp;
    private Tensor? _normW, _normB, _norm2W, _norm2B;

    public ZonosBlock(ZonosConfig cfg)
    {
        _cfg = cfg;
        _attn = new DiaAttention(cfg.NumHeads, cfg.NumKvHeads, cfg.HeadDim, cfg.Hidden, cfg.Hidden, cfg.Hidden, useRope: true);
        _mlp = new DiaMlp(cfg.Hidden, cfg.FfnIntermediate);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
        _norm2W = WhisperOps.EnsureF32(w[$"{prefix}.norm2.weight"]);
        _norm2B = WhisperOps.EnsureF32(w[$"{prefix}.norm2.bias"]);

        // Split fused in_proj [q+kv+kv, hidden] → q/k/v for DiaAttention; map out_proj → o_proj.
        Tensor inProj = WhisperOps.EnsureF32(w[$"{prefix}.mixer.in_proj.weight"]);
        int qDim = _cfg.NumHeads * _cfg.HeadDim, kvDim = _cfg.NumKvHeads * _cfg.HeadDim, h = _cfg.Hidden;
        Dictionary<string, Tensor> attnW = new()
        {
            ["a.q_proj.weight"] = SliceRows(inProj, 0, qDim, h),
            ["a.k_proj.weight"] = SliceRows(inProj, qDim, kvDim, h),
            ["a.v_proj.weight"] = SliceRows(inProj, qDim + kvDim, kvDim, h),
            ["a.o_proj.weight"] = WhisperOps.EnsureF32(w[$"{prefix}.mixer.out_proj.weight"]),
        };
        _attn.LoadWeights(attnW, "a");

        Dictionary<string, Tensor> mlpW = new()
        {
            ["m.gate_up_proj.weight"] = WhisperOps.EnsureF32(w[$"{prefix}.mlp.fc1.weight"]),
            ["m.down_proj.weight"] = WhisperOps.EnsureF32(w[$"{prefix}.mlp.fc2.weight"]),
        };
        _mlp.LoadWeights(mlpW, "m");
    }

    public Tensor Forward(IBackend backend, Tensor x, int t, int posStart, StreamingKvCache cache, int layer,
        Tensor? mask, float[] cos, float[] sin)
    {
        int h = _cfg.Hidden;
        Tensor pre = new(new TensorShape(1, t, h), DType.F32);
        backend.LayerNorm(pre, x, _normW!, _normB!, _cfg.NormEps);
        Tensor attn = _attn.SelfForward(backend, pre, t, posStart, cache, layer, mask, cos, sin);
        pre.Dispose();
        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.Add(afterAttn, x, attn); attn.Dispose();

        Tensor pre2 = new(new TensorShape(1, t, h), DType.F32);
        backend.LayerNorm(pre2, afterAttn, _norm2W!, _norm2B!, _cfg.NormEps);
        Tensor mlp = _mlp.Forward(backend, pre2, t); pre2.Dispose();
        Tensor outT = new(x.Shape, DType.F32);
        backend.Add(outT, afterAttn, mlp); afterAttn.Dispose(); mlp.Dispose();
        return outT;
    }

    private static Tensor SliceRows(Tensor src, int rowStart, int rowCount, int cols)
    {
        Tensor dst = new(new TensorShape(rowCount, cols), DType.F32);
        Buffer.MemoryCopy((float*)src.DataPointer + (long)rowStart * cols, (void*)dst.DataPointer,
            (long)rowCount * cols * 4, (long)rowCount * cols * 4);
        return dst;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] norms = [_normW, _normB, _norm2W, _norm2B];
        foreach (Tensor? t in norms) if (t is not null) yield return t;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
    }
}
