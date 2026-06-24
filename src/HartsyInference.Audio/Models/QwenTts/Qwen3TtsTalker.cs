using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Qwen3-TTS talker: the autoregressive backbone that emits codebook-0 (semantic) tokens at 12.5 Hz.
/// It carries dual embedding tables — <c>text_embedding</c> (151936) and <c>codec_embedding</c> (3072) — fuses a
/// text token (projected through the <c>text_projection</c> MLP) and the previous codec token into one
/// <c>[1,1,hidden]</c> step embedding, runs the headless <see cref="Qwen3Model"/> (per-head q/k-norm + GQA +
/// 3D interleaved mRoPE), and projects the final hidden through <c>codec_head</c> (2048→3072) to logits for
/// codebook 0. The MTP code-predictor consumes the same final hidden to fill codebooks 1..15.
///
/// <para>For pure-text TTS the talker uses the running text position for all three mRoPE axes, which the
/// headless <see cref="Qwen3Model"/> realizes with its interleaved RoPE; <see cref="Qwen3Mrope"/> is the
/// documented entry point for the multimodal-axis case.</para></summary>
public sealed unsafe class Qwen3TtsTalker : IDisposable
{
    private readonly Qwen3TtsConfig _cfg;
    private readonly Qwen3Model _backbone;
    private readonly Qwen3Mrope _mrope;
    private Tensor? _textEmbedding;     // [TextVocab, hidden]
    private Tensor? _codecEmbedding;    // [CodecVocab, hidden]
    private Tensor? _textProjW1, _textProjB1, _textProjW2, _textProjB2;   // text_projection MLP
    private Tensor? _codecHeadW, _codecHeadB;   // [CodecHeadOut, hidden]
    private int _disposed;

    public Qwen3TtsConfig Config => _cfg;
    public Qwen3Model Backbone => _backbone;
    public int HiddenSize => _cfg.Talker.HiddenSize;

    public Qwen3TtsTalker(Qwen3TtsConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen3Model(cfg.Talker);
        _mrope = new Qwen3Mrope(cfg.Talker.HeadDim, cfg.Talker.RopeTheta, cfg.MropeSection);
    }

    public Qwen3Mrope Mrope => _mrope;

    /// <summary>Loads the talker: dual embeddings, the text-projection MLP, the codec head, and the headless
    /// Qwen3 body under <paramref name="backbonePrefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string backbonePrefix = "talker.model",
        string talkerPrefix = "talker")
    {
        // Real Qwen3-TTS layout: the dual embeddings + final norm + decoder layers live under `talker.model.*`,
        // while the codec head and the text-projection MLP (named linear_fc1/linear_fc2) sit one level up under
        // `talker.*`.
        _textEmbedding = WhisperOps.EnsureF32(w[$"{backbonePrefix}.text_embedding.weight"]);
        _codecEmbedding = WhisperOps.EnsureF32(w[$"{backbonePrefix}.codec_embedding.weight"]);
        _textProjW1 = WhisperOps.EnsureF32(w[$"{talkerPrefix}.text_projection.linear_fc1.weight"]);
        _textProjB1 = TryGet(w, $"{talkerPrefix}.text_projection.linear_fc1.bias");
        _textProjW2 = WhisperOps.EnsureF32(w[$"{talkerPrefix}.text_projection.linear_fc2.weight"]);
        _textProjB2 = TryGet(w, $"{talkerPrefix}.text_projection.linear_fc2.bias");
        _codecHeadW = WhisperOps.EnsureF32(w[$"{talkerPrefix}.codec_head.weight"]);
        _codecHeadB = TryGet(w, $"{talkerPrefix}.codec_head.bias");
        _backbone.LoadWeightsHeadless(w, backbonePrefix);
    }

    /// <summary>Builds the per-step fused embedding <c>[1,1,hidden]</c> from a text token and the previous codec
    /// token. Text tokens project through the <c>text_projection</c> MLP; codec tokens use the raw codec table.
    /// A negative id contributes nothing (used to hold one stream while the other advances).</summary>
    public Tensor EmbedStep(IBackend backend, int textToken, int codecToken)
    {
        int h = _cfg.Talker.HiddenSize;
        Tensor embed = new(new TensorShape(1, 1, h), DType.F32);
        float* dst = (float*)embed.DataPointer;     // freshly allocated, zero-initialized
        if (textToken >= 0)
        {
            Tensor raw = new(new TensorShape(1, 1, h), DType.F32);
            float* rp = (float*)raw.DataPointer;
            float* tp = (float*)_textEmbedding!.DataPointer + (long)textToken * h;
            Buffer.MemoryCopy(tp, rp, h * 4, h * 4);
            Tensor projected = ProjectText(backend, raw);
            raw.Dispose();
            float* pp = (float*)projected.DataPointer;
            for (int i = 0; i < h; i++) dst[i] += pp[i];
            projected.Dispose();
        }
        if (codecToken >= 0)
        {
            float* cp = (float*)_codecEmbedding!.DataPointer + (long)codecToken * h;
            for (int i = 0; i < h; i++) dst[i] += cp[i];
        }
        return embed;
    }

    /// <summary>Runs the headless backbone over a prepared <c>[1,t,hidden]</c> embedding block and returns the
    /// final-norm hidden <c>[1,t,hidden]</c>. Callers feed prefill in one shot then single steps.</summary>
    public Tensor Forward(IBackend backend, Tensor embeds, int t, int posStart, StreamingKvCache cache)
    {
        return _backbone.ForwardEmbeds(backend, embeds, t, posStart, cache);
    }

    /// <summary>Projects the final hidden <c>[1,1,hidden]</c> through <c>codec_head</c> to codebook-0 logits
    /// <c>[CodecHeadOut]</c>, then samples one token with repetition handling. Returns the sampled codebook-0
    /// token (a value in codec space; <see cref="Qwen3TtsConfig.CodecEos"/> signals stop).</summary>
    public int SampleCodebook0(IBackend backend, Tensor finalHidden, ref uint rng, ReadOnlySpan<int> recent)
    {
        int h = _cfg.Talker.HiddenSize;
        Tensor logits = WhisperOps.ProjectLinear(backend, finalHidden, _codecHeadW!, _codecHeadB, 1, 1, h, _cfg.CodecHeadOut);
        ApplyRepetitionPenalty(logits, _cfg.CodecHeadOut, recent, _cfg.RepetitionPenalty);
        int tok = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _cfg.CodecHeadOut),
            _cfg.CodecHeadOut, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
        logits.Dispose();
        return tok;
    }

    /// <summary>Returns the talker's codec embedding row for <paramref name="codecToken"/> as <c>[1,1,h]</c>.
    /// The code predictor embeds codebook-0 through this table (then projects to MTP width).</summary>
    public Tensor CodecEmbedRow(IBackend backend, int codecToken)
    {
        int h = _cfg.Talker.HiddenSize;
        Tensor row = new(new TensorShape(1, 1, h), DType.F32);
        if (codecToken >= 0)
        {
            float* src = (float*)_codecEmbedding!.DataPointer + (long)codecToken * h;
            Buffer.MemoryCopy(src, (void*)row.DataPointer, h * 4, h * 4);
        }
        return row;
    }

    /// <summary>Adds the talker's codebook-0 codec embedding row to <paramref name="dst"/> (channels <c>[h]</c>).
    /// Used by the per-frame talker feedback (codebook-0 part of the 16-code sum).</summary>
    public void AddCodecEmbed(Tensor dst, int codecToken)
    {
        if (codecToken < 0) return;
        int h = _cfg.Talker.HiddenSize;
        float* d = (float*)dst.DataPointer;
        float* cp = (float*)_codecEmbedding!.DataPointer + (long)codecToken * h;
        for (int i = 0; i < h; i++) d[i] += cp[i];
    }

    private Tensor ProjectText(IBackend backend, Tensor x)
    {
        int h = _cfg.Talker.HiddenSize;
        int mid = (int)_textProjW1!.Shape[0];
        Tensor a = WhisperOps.ProjectLinear(backend, x, _textProjW1!, _textProjB1, 1, 1, h, mid);
        backend.Silu(a, a);   // Qwen3-TTS text_projection is fc1 -> SiLU -> fc2 (verified vs reference)
        Tensor o = WhisperOps.ProjectLinear(backend, a, _textProjW2!, _textProjB2, 1, 1, mid, h);
        a.Dispose();
        return o;
    }

    private static void ApplyRepetitionPenalty(Tensor logits, int count, ReadOnlySpan<int> recent, float penalty)
    {
        if (penalty == 1f || recent.Length == 0) return;
        float* p = (float*)logits.DataPointer;
        for (int i = 0; i < recent.Length; i++)
        {
            int t = recent[i];
            if ((uint)t >= (uint)count) continue;
            p[t] = p[t] > 0 ? p[t] / penalty : p[t] * penalty;
        }
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_textEmbedding, _codecEmbedding, _textProjW1, _textProjB1, _textProjW2, _textProjB2, _codecHeadW, _codecHeadB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
