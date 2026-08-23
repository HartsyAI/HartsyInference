using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Qwen3-TTS talker: the autoregressive backbone that emits codebook-0 (semantic) tokens at 12.5 Hz.
/// It carries dual embedding tables — <c>text_embedding</c> (151936) and <c>codec_embedding</c> (3072) — fuses a
/// text token (projected through the <c>text_projection</c> MLP) and the previous codec token into one
/// <c>[1,1,hidden]</c> step embedding, runs the headless <see cref="Qwen3Model"/> (per-head q/k-norm + GQA +
/// 3D interleaved mRoPE), and projects the final hidden through <c>codec_head</c> (hidden→3072) to logits for
/// codebook 0. The MTP code-predictor consumes the same final hidden to fill codebooks 1..15. Text embeddings
/// are <c>text_hidden_size</c> wide (2048 on BOTH sizes; the 0.6B talker hidden is 1024) and reach hidden width
/// only through <c>text_projection</c>.
///
/// <para>For pure-text TTS the talker uses the running text position for all three mRoPE axes, which the
/// headless <see cref="Qwen3Model"/> realizes with its interleaved RoPE; <see cref="Qwen3Mrope"/> is the
/// documented entry point for the multimodal-axis case.</para></summary>
public sealed unsafe class Qwen3TtsTalker : IDisposable
{
    private readonly Qwen3TtsConfig _cfg;
    private readonly Qwen3Model _backbone;
    private readonly Qwen3Mrope _mrope;
    private Tensor? _textEmbedding;     // [TextVocab, textHidden] — textHidden stays 2048 even on the 0.6B (hidden 1024)
    private Tensor? _codecEmbedding;    // [CodecVocab, hidden]
    private Tensor? _textProjW1, _textProjB1, _textProjW2, _textProjB2;   // text_projection MLP: textHidden -> textHidden -> hidden
    private Tensor? _codecHeadW, _codecHeadB;   // [CodecHeadOut, hidden]
    private int _textHidden, _textProjMid;      // derived from the checkpoint at load
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
        // text_hidden_size is decoupled from hidden_size (0.6B: text 2048, hidden 1024); derive both text-side
        // widths from the checkpoint and guard every stride the embed/projection paths depend on.
        _textHidden = (int)_textEmbedding.Shape[1];
        _textProjMid = (int)_textProjW1.Shape[0];
        int h = _cfg.Talker.HiddenSize;
        RequireDim(_textEmbedding, 0, _cfg.TextVocabSize, "text_embedding rows");
        RequireDim(_textProjW1, 1, _textHidden, "text_projection fc1 in-dim");
        RequireDim(_textProjW2, 1, _textProjMid, "text_projection fc2 in-dim");
        RequireDim(_textProjW2, 0, h, "text_projection fc2 out-dim");
        RequireDim(_codecEmbedding, 0, _cfg.CodecVocabSize, "codec_embedding rows");
        RequireDim(_codecEmbedding, 1, h, "codec_embedding dim");
        RequireDim(_codecHeadW, 0, _cfg.CodecHeadOut, "codec_head out-dim");
        RequireDim(_codecHeadW, 1, h, "codec_head in-dim");
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
            Tensor raw = new(new TensorShape(1, 1, _textHidden), DType.F32);
            float* rp = (float*)raw.DataPointer;
            float* tp = (float*)_textEmbedding!.DataPointer + (long)textToken * _textHidden;
            Buffer.MemoryCopy(tp, rp, _textHidden * 4L, _textHidden * 4L);
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
    public Tensor Forward(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache)
    {
        return _backbone.ForwardEmbeds(backend, embeds, t, posStart, cache);
    }

    /// <summary>Projects the final hidden <c>[1,1,hidden]</c> through <c>codec_head</c> to codebook-0 logits
    /// <c>[CodecHeadOut]</c>, then samples one token faithfully to the reference generate() setup: repetition
    /// penalty over <paramref name="generated"/> (every previously emitted codebook-0 token), all control ids
    /// (&gt;= <see cref="Qwen3TtsConfig.CodecRealVocab"/>) suppressed EXCEPT <see cref="Qwen3TtsConfig.CodecEos"/>,
    /// and the eos itself suppressed while <paramref name="allowEos"/> is false (min_new_tokens). Returns the
    /// sampled codebook-0 token (<see cref="Qwen3TtsConfig.CodecEos"/> signals stop).</summary>
    public int SampleCodebook0(IBackend backend, Tensor finalHidden, ref uint rng, ReadOnlySpan<int> generated,
        bool allowEos = true)
    {
        int h = _cfg.Talker.HiddenSize;
        Tensor logits = WhisperOps.ProjectLinear(backend, finalHidden, _codecHeadW!, _codecHeadB, 1, 1, h, _cfg.CodecHeadOut);
        float* p = (float*)logits.DataPointer;
        ApplyRepetitionPenalty(logits, _cfg.CodecHeadOut, generated, _cfg.RepetitionPenalty);
        for (int i = _cfg.CodecRealVocab; i < _cfg.CodecHeadOut; i++)
        {
            if (i != _cfg.CodecEos) p[i] = float.NegativeInfinity;   // reference suppress_tokens list
        }
        if (!allowEos && (uint)_cfg.CodecEos < (uint)_cfg.CodecHeadOut) p[_cfg.CodecEos] = float.NegativeInfinity;
        int tok = NucleusSampler.Draw(new Span<float>(p, _cfg.CodecHeadOut),
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

    private Tensor ProjectText(IBackend backend, Tensor x)
    {
        Tensor a = WhisperOps.ProjectLinear(backend, x, _textProjW1!, _textProjB1, 1, 1, _textHidden, _textProjMid);
        backend.Silu(a, a);   // Qwen3-TTS text_projection is fc1 -> SiLU -> fc2 (verified vs reference)
        Tensor o = WhisperOps.ProjectLinear(backend, a, _textProjW2!, _textProjB2, 1, 1, _textProjMid, _cfg.Talker.HiddenSize);
        a.Dispose();
        return o;
    }

    private static void ApplyRepetitionPenalty(Tensor logits, int count, ReadOnlySpan<int> generated, float penalty)
    {
        if (penalty == 1f || generated.Length == 0) return;
        float* p = (float*)logits.DataPointer;
        for (int i = 0; i < generated.Length; i++)
        {
            int t = generated[i];
            if ((uint)t >= (uint)count) continue;
            p[t] = p[t] > 0 ? p[t] / penalty : p[t] * penalty;
        }
    }

    private static void RequireDim(Tensor? t, int axis, int expected, string what)
    {
        if (t is not null && (int)t.Shape[axis] != expected)
        {
            throw new InvalidOperationException(
                $"Qwen3-TTS talker config mismatch: {what} is {t.Shape[axis]} in the checkpoint but the config says {expected}.");
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
