using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Music;

/// <summary>ACE-Step v1.5's <c>AceStepEncoderLayer</c> — a bidirectional Qwen3 block (pre-norm GQA self-attention
/// with RoPE + pre-norm SwiGLU MLP, plain residuals) used by the lyric and timbre condition encoders. Layers
/// alternate sliding(128)/full attention via <c>AceStep15Config.IsSlidingLayer</c>.</summary>
internal sealed class AceStep15EncoderLayer
{
    private readonly AceStep15Config _c;
    private readonly AceStep15Attention _attn;
    private readonly SwiGluFfn _mlp;
    private Tensor? _inputNorm, _postAttnNorm;

    public AceStep15EncoderLayer(AceStep15Config config)
    {
        _c = config;
        _attn = new AceStep15Attention(config);
        _mlp = new SwiGluFfn(config.HiddenSize, config.IntermediateSize);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inputNorm = EnsureF32(w[$"{prefix}.input_layernorm.weight"]);
        _postAttnNorm = EnsureF32(w[$"{prefix}.post_attention_layernorm.weight"]);
        _attn.LoadWeights(w, $"{prefix}.self_attn");
        _mlp.LoadSwiGluWeights(
            w[$"{prefix}.mlp.gate_proj.weight"], null,
            w[$"{prefix}.mlp.up_proj.weight"], null,
            w[$"{prefix}.mlp.down_proj.weight"], null);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inputNorm is not null) yield return _inputNorm;
        if (_postAttnNorm is not null) yield return _postAttnNorm;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
    }

    /// <summary>One block over <c>[1, S, H]</c>; returns a fresh tensor (input untouched).</summary>
    public Tensor Forward(IBackend backend, Tensor x, float[] ropeCos, float[] ropeSin, Tensor? mask)
    {
        int s = (int)x.Shape[1];
        TensorShape shape = new TensorShape(1, s, _c.HiddenSize);

        Tensor normed = new Tensor(shape, DType.F32);
        backend.RmsNorm(normed, x, _inputNorm!, _c.RmsNormEps);
        Tensor attn = _attn.Forward(backend, normed, crossKv: null, ropeCos, ropeSin, mask);
        normed.Dispose();
        Tensor afterAttn = new Tensor(shape, DType.F32);
        backend.Add(afterAttn, x, attn);
        attn.Dispose();

        Tensor preMlp = new Tensor(shape, DType.F32);
        backend.RmsNorm(preMlp, afterAttn, _postAttnNorm!, _c.RmsNormEps);
        Tensor mlpOut = _mlp.Forward(backend, preMlp, 1, s);
        preMlp.Dispose();
        Tensor result = new Tensor(shape, DType.F32);
        backend.Add(result, afterAttn, mlpOut);
        afterAttn.Dispose();
        mlpOut.Dispose();
        return result;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
