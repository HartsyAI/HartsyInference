using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.LanguageModels.Gpt;

/// <summary>GPT-2-style pre-norm Transformer backbone (learned absolute positions, multi-head attention,
/// 4× GELU MLP, LayerNorm, <c>bias=False</c> on linears). The shared decoder body for the 2019-vintage
/// GPT-2 audio LMs in the package (Bark's three stages; reusable by XTTS / ChatTTS) — parameterized by
/// depth / width / head count rather than copied per model. Operates on caller-supplied input embeddings
/// <c>[1, T, hidden]</c> (each model owns its own token / codebook embedding tables + output heads),
/// adds the learned positional embedding, runs the block stack, returns the final-LayerNorm hidden state.
/// Supports causal and non-causal (full bidirectional, e.g. Bark-Fine). Runs full-sequence (no KV cache);
/// AR callers re-feed the growing prefix each step — simple + correct, perf-tunable later.</summary>
public sealed unsafe class GptBackbone : IDisposable
{
    private readonly GptConfig _cfg;
    private readonly GptBlock[] _blocks;
    private int _disposed;

    private Tensor? _posEmbed;     // [blockSize, hidden] learned absolute positions
    private Tensor? _lnFGamma, _lnFBeta;

    public GptConfig Config => _cfg;

    public GptBackbone(GptConfig cfg)
    {
        _cfg = cfg;
        _blocks = new GptBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new GptBlock(cfg);
    }

    /// <summary>Loads the positional embedding, blocks, and final LayerNorm. Keys follow the HF Bark
    /// scheme by default (<c>position_embeds_layer.weight</c>, <c>layers.{i}.*</c>, <c>layernorm_final.*</c>);
    /// pass the model's own keys.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string posKey, string blockPrefix,
        string lnFGammaKey, string lnFBetaKey)
    {
        _posEmbed = WhisperOps.EnsureF32(w[posKey]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{blockPrefix}.{i}");
        _lnFGamma = WhisperOps.EnsureF32(w[lnFGammaKey]);
        _lnFBeta = WhisperOps.EnsureF32(w[lnFBetaKey]);
    }

    /// <summary>Runs the stack over <paramref name="inputEmbeds"/> <c>[1, T, hidden]</c>, adding learned
    /// positions from index 0. Causal unless <paramref name="nonCausal"/>. Returns <c>[1, T, hidden]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor inputEmbeds, bool nonCausal = false)
    {
        ThrowIfDisposed();
        int t = (int)inputEmbeds.Shape[1];
        int h = _cfg.Hidden;

        Tensor hidden = new(inputEmbeds.Shape, DType.F32);
        float* ip = (float*)inputEmbeds.DataPointer;
        float* op = (float*)hidden.DataPointer;
        float* pe = (float*)_posEmbed!.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long off = (long)s * h;
            for (int c = 0; c < h; c++) op[off + c] = ip[off + c] + pe[off + c];
        }

        Tensor? causalMask = (!nonCausal && t > 1) ? BuildCausalMask(t) : null;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, hidden, causalMask);
            hidden.Dispose();
            hidden = next;
        }
        causalMask?.Dispose();

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnFGamma!, _lnFBeta!, 1e-5f);
        hidden.Dispose();
        return normed;
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new(new TensorShape(1, 1, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                mp[(long)q * t + k] = k <= q ? 0f : float.NegativeInfinity;
        return mask;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_posEmbed is not null) yield return _posEmbed;
        foreach (GptBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_lnFGamma is not null) yield return _lnFGamma;
        if (_lnFBeta is not null) yield return _lnFBeta;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (GptBlock b in _blocks) b.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(GptBackbone));
    }
}
