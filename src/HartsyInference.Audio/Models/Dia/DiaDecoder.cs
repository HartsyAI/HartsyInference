using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Dia decoder: 9 per-codebook embedding tables (summed) + N pre-norm layers (causal GQA
/// self-attention with RoPE + cross-attention to the encoder + SwiGLU MLP) + final RMSNorm + a fused
/// <c>logits_dense</c> head producing all 9 codebooks' logits per step. Encoder K/V are precomputed once
/// via <see cref="PrecomputeCrossKv"/>; self-attention uses a <see cref="StreamingKvCache"/>.</summary>
public sealed unsafe class DiaDecoder : IDisposable
{
    private readonly DiaConfig _cfg;
    private readonly DiaDecoderLayer[] _layers;
    private readonly Tensor?[] _channelEmbed;     // [Channels] each [AudioVocab, DecoderDim]
    private Tensor? _finalNorm, _logitsW;
    private float[]? _cos, _sin;
    private int _disposed;

    public DiaDecoder(DiaConfig cfg)
    {
        _cfg = cfg;
        _layers = new DiaDecoderLayer[cfg.DecoderLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new DiaDecoderLayer(cfg);
        _channelEmbed = new Tensor?[cfg.Channels];
    }

    public int NumLayers => _cfg.DecoderLayers;
    public int KvHeads => _cfg.DecoderSelfKvHeads;
    public int HeadDim => _cfg.HeadDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.decoder",
        string logitsKey = "logits_dense.weight")
    {
        for (int c = 0; c < _cfg.Channels; c++)
            _channelEmbed[c] = WhisperOps.EnsureF32(w[$"{prefix}.embeddings.{c}.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
        _finalNorm = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _logitsW = WhisperOps.EnsureF32(w[logitsKey]);
    }

    /// <summary>Projects + caches the encoder context as cross-attention K/V in every layer.</summary>
    public void PrecomputeCrossKv(IBackend backend, Tensor encOut, int encLen)
    {
        foreach (DiaDecoderLayer l in _layers) l.PrecomputeCrossKv(backend, encOut, encLen);
        _cos ??= RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, _cfg.MaxAudio).Cos;
        _sin ??= RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, _cfg.MaxAudio).Sin;
    }

    /// <summary>One decode step over the current 9 channel tokens → logits <c>[Channels, AudioVocab]</c>.</summary>
    public Tensor StepLogits(IBackend backend, ReadOnlySpan<int> channelTokens, int posStart, StreamingKvCache cache)
    {
        int dim = _cfg.DecoderDim;
        Tensor x = new(new TensorShape(1, 1, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < _cfg.Channels; c++)
        {
            float* row = (float*)_channelEmbed[c]!.DataPointer + (long)channelTokens[c] * dim;
            for (int i = 0; i < dim; i++) xp[i] += row[i];
        }

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, posStart, cache, i, _cos!, _sin!);
            x.Dispose();
            x = next;
        }
        // Every layer appended its K/V at the shared CurrentLength; advance once per step so the next step's
        // self-attention sees this position (Append does not advance the counter itself).
        cache.AdvanceLength(1);
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, _finalNorm!, _cfg.NormEps); x.Dispose();

        // Fused head: [1,1,dim] → [1,1,Channels*AudioVocab].
        Tensor logits = WhisperOps.ProjectLinear(backend, normed, _logitsW!, null, 1, 1, dim,
            _cfg.Channels * _cfg.AudioVocab);
        normed.Dispose();
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _channelEmbed) if (e is not null) yield return e;
        foreach (DiaDecoderLayer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_finalNorm is not null) yield return _finalNorm;
        if (_logitsW is not null) yield return _logitsW;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _channelEmbed.Length; i++) _channelEmbed[i] = null;
        _finalNorm = null; _logitsW = null;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One Dia decoder block: causal GQA self-attn + cross-attn + SwiGLU, each pre-normed and residual.</summary>
public sealed unsafe class DiaDecoderLayer
{
    private readonly DiaConfig _cfg;
    private readonly DiaAttention _self;
    private readonly DiaAttention _cross;
    private readonly DiaMlp _mlp;
    private Tensor? _preSa, _preCa, _preMlp;

    public DiaDecoderLayer(DiaConfig cfg)
    {
        _cfg = cfg;
        _self = new DiaAttention(cfg.DecoderSelfQHeads, cfg.DecoderSelfKvHeads, cfg.HeadDim,
            cfg.DecoderDim, cfg.DecoderDim, cfg.DecoderDim, useRope: true);
        _cross = new DiaAttention(cfg.DecoderCrossQHeads, cfg.DecoderCrossKvHeads, cfg.HeadDim,
            cfg.DecoderDim, cfg.EncoderDim, cfg.DecoderDim, useRope: false);
        _mlp = new DiaMlp(cfg.DecoderDim, cfg.DecoderFfn);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _preSa = WhisperOps.EnsureF32(w[$"{prefix}.pre_sa_norm.weight"]);
        _preCa = WhisperOps.EnsureF32(w[$"{prefix}.pre_ca_norm.weight"]);
        _preMlp = WhisperOps.EnsureF32(w[$"{prefix}.pre_mlp_norm.weight"]);
        _self.LoadWeights(w, $"{prefix}.self_attention");
        _cross.LoadWeights(w, $"{prefix}.cross_attention");
        _mlp.LoadWeights(w, $"{prefix}.mlp");
    }

    public void PrecomputeCrossKv(IBackend backend, Tensor encOut, int encLen)
        => _cross.PrecomputeCrossKv(backend, encOut, encLen);

    public Tensor Forward(IBackend backend, Tensor x, int posStart, StreamingKvCache cache, int layerIndex,
        float[] cos, float[] sin)
    {
        int dim = _cfg.DecoderDim;
        TensorShape shape = new(1, 1, dim);

        Tensor preSa = new(shape, DType.F32);
        backend.RmsNorm(preSa, x, _preSa!, _cfg.NormEps);
        Tensor sa = _self.SelfForward(backend, preSa, 1, posStart, cache, layerIndex, null, cos, sin);
        preSa.Dispose();
        Tensor afterSa = new(shape, DType.F32);
        backend.Add(afterSa, x, sa); sa.Dispose();

        Tensor preCa = new(shape, DType.F32);
        backend.RmsNorm(preCa, afterSa, _preCa!, _cfg.NormEps);
        Tensor ca = _cross.CrossForward(backend, preCa, 1);
        preCa.Dispose();
        Tensor afterCa = new(shape, DType.F32);
        backend.Add(afterCa, afterSa, ca); afterSa.Dispose(); ca.Dispose();

        Tensor preMlp = new(shape, DType.F32);
        backend.RmsNorm(preMlp, afterCa, _preMlp!, _cfg.NormEps);
        Tensor mlp = _mlp.Forward(backend, preMlp, 1); preMlp.Dispose();
        Tensor outT = new(shape, DType.F32);
        backend.Add(outT, afterCa, mlp); afterCa.Dispose(); mlp.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_preSa is not null) yield return _preSa;
        if (_preCa is not null) yield return _preCa;
        if (_preMlp is not null) yield return _preMlp;
        foreach (Tensor t in _self.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cross.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
    }
}
