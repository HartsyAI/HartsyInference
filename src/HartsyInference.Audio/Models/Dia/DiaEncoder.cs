using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Dia text encoder: byte embedding + N pre-norm transformer layers (non-causal MHA self-attention
/// with RoPE + SwiGLU MLP) + final RMSNorm. Produces the <c>[1, T, EncoderDim]</c> context the decoder
/// cross-attends to.</summary>
public sealed unsafe class DiaEncoder : IDisposable
{
    private readonly DiaConfig _cfg;
    private readonly DiaEncoderLayer[] _layers;
    private Tensor? _embed, _finalNorm;
    private int _disposed;

    public DiaEncoder(DiaConfig cfg)
    {
        _cfg = cfg;
        _layers = new DiaEncoderLayer[cfg.EncoderLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new DiaEncoderLayer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.encoder")
    {
        _embed = WhisperOps.EnsureF32(w[$"{prefix}.embedding.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
        _finalNorm = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
    }

    /// <summary>Encodes byte token ids <paramref name="tokens"/> → <c>[1, T, EncoderDim]</c>.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokens)
    {
        if (_embed is null) throw new InvalidOperationException("DiaEncoder not loaded.");
        int t = tokens.Length, dim = _cfg.EncoderDim;
        Tensor x = new(new TensorShape(1, t, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* ep = (float*)_embed.DataPointer;
        for (int s = 0; s < t; s++)
            Buffer.MemoryCopy(ep + (long)tokens[s] * dim, xp + (long)s * dim, dim * 4, dim * 4);

        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, Math.Max(1, t));
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, t, cos, sin);
            x.Dispose();
            x = next;
        }
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, _finalNorm!, _cfg.NormEps);
        x.Dispose();
        return normed;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embed is not null) yield return _embed;
        foreach (DiaEncoderLayer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_finalNorm is not null) yield return _finalNorm;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _embed = null; _finalNorm = null;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One Dia encoder block: <c>x += attn(pre_sa_norm(x)); x += mlp(post_sa_norm(x))</c>.</summary>
public sealed unsafe class DiaEncoderLayer
{
    private readonly DiaConfig _cfg;
    private readonly DiaAttention _attn;
    private readonly DiaMlp _mlp;
    private Tensor? _preNorm, _postNorm;

    public DiaEncoderLayer(DiaConfig cfg)
    {
        _cfg = cfg;
        _attn = new DiaAttention(cfg.EncoderHeads, cfg.EncoderKvHeads, cfg.HeadDim,
            cfg.EncoderDim, cfg.EncoderDim, cfg.EncoderDim, useRope: true);
        _mlp = new DiaMlp(cfg.EncoderDim, cfg.EncoderFfn);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _preNorm = WhisperOps.EnsureF32(w[$"{prefix}.pre_sa_norm.weight"]);
        _postNorm = WhisperOps.EnsureF32(w[$"{prefix}.post_sa_norm.weight"]);
        _attn.LoadWeights(w, $"{prefix}.self_attention");
        _mlp.LoadWeights(w, $"{prefix}.mlp");
    }

    public Tensor Forward(IBackend backend, Tensor x, int t, float[] cos, float[] sin)
    {
        int dim = _cfg.EncoderDim;
        Tensor pre = new(new TensorShape(1, t, dim), DType.F32);
        backend.RmsNorm(pre, x, _preNorm!, _cfg.NormEps);
        Tensor attn = _attn.SelfForward(backend, pre, t, 0, null, 0, null, cos, sin);
        pre.Dispose();
        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.Add(afterAttn, x, attn); attn.Dispose();

        Tensor pre2 = new(new TensorShape(1, t, dim), DType.F32);
        backend.RmsNorm(pre2, afterAttn, _postNorm!, _cfg.NormEps);
        Tensor mlp = _mlp.Forward(backend, pre2, t); pre2.Dispose();
        Tensor outT = new(x.Shape, DType.F32);
        backend.Add(outT, afterAttn, mlp);
        afterAttn.Dispose(); mlp.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_preNorm is not null) yield return _preNorm;
        if (_postNorm is not null) yield return _postNorm;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
    }
}
