using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>One Qwen2 transformer block — RMSNorm + self-attention + residual + RMSNorm
/// + SwiGLU MLP + residual. Pre-norm (Llama / Qwen) order, not Gemma's sandwich norm.</summary>
internal sealed unsafe class Qwen2DecoderLayer
{
    private readonly Qwen2Config _cfg;
    private readonly Qwen2Attention _attn;
    private readonly Qwen2Mlp _mlp;

    private Tensor? _inputNorm;
    private Tensor? _postAttnNorm;

    public Qwen2DecoderLayer(Qwen2Config cfg, Qwen2RotaryEmbedding rope, int layerIndex)
    {
        _cfg = cfg;
        _attn = new Qwen2Attention(cfg, rope, layerIndex);
        _mlp = new Qwen2Mlp(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inputNorm = WhisperOps.EnsureF32(w[$"{prefix}.input_layernorm.weight"]);
        _postAttnNorm = WhisperOps.EnsureF32(w[$"{prefix}.post_attention_layernorm.weight"]);
        _attn.LoadWeights(w, $"{prefix}.self_attn");
        _mlp.LoadWeights(w, $"{prefix}.mlp");
    }

    public Tensor Forward(IBackend backend, Tensor hidden, int batch, int t, int posStart,
        StreamingKvCache cache, Tensor? causalMask)
    {
        int h = _cfg.HiddenSize;
        TensorShape shape = new(batch, t, h);

        // Pre-attention RMSNorm.
        Tensor preAttn = new(shape, DType.F32);
        backend.RmsNorm(preAttn, hidden, _inputNorm!, _cfg.RmsNormEps);

        // Attention.
        Tensor attnOut = _attn.Forward(backend, preAttn, batch, t, posStart, cache, causalMask);
        preAttn.Dispose();

        // Residual.
        Tensor afterAttn = new(shape, DType.F32);
        backend.Add(afterAttn, hidden, attnOut);
        attnOut.Dispose();

        // Pre-MLP RMSNorm.
        Tensor preMlp = new(shape, DType.F32);
        backend.RmsNorm(preMlp, afterAttn, _postAttnNorm!, _cfg.RmsNormEps);

        // MLP.
        Tensor mlpOut = _mlp.Forward(backend, preMlp, batch, t);
        preMlp.Dispose();

        // Residual.
        Tensor result = new(shape, DType.F32);
        backend.Add(result, afterAttn, mlpOut);
        afterAttn.Dispose();
        mlpOut.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inputNorm is not null) yield return _inputNorm;
        if (_postAttnNorm is not null) yield return _postAttnNorm;
        foreach (Tensor t in _attn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
    }
}
