using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step v1.5's <c>AceStepConditionEncoder</c> — builds the packed cross-attention sequence for the DiT:
/// <c>text_projector</c> (Linear 1024→2048, no bias) over Qwen3-Embedding-0.6B prompt states, the lyric encoder
/// (Linear 1024→2048 + 8 bidirectional Qwen3 blocks + RMSNorm) over Qwen3-Embedding lyric states, and the optional
/// timbre encoder (Linear 64→2048 + prepended CLS <c>special_token</c> + 4 blocks + RMSNorm, CLS output kept) over
/// reference-audio Oobleck latents. The reference packs valid tokens in order <b>[lyric, timbre, text]</b>
/// (<c>pack_sequences(lyric, timbre)</c> then <c>pack_sequences(·, text)</c>); with batch 1 and no padding that is a
/// plain concat. Null lyric/timbre inputs contribute zero tokens — the reference reaches the same state via all-zero
/// masks (unconditional drops use <c>null_condition_emb</c>, which turbo never needs: no CFG).</summary>
public sealed unsafe class AceStep15ConditionEncoder
{
    private readonly AceStep15Config _c;
    private readonly AceStep15EncoderLayer[] _lyricLayers;
    private readonly AceStep15EncoderLayer[] _timbreLayers;
    private Tensor? _textProjW;
    private Tensor? _lyricEmbedW, _lyricEmbedB, _lyricNorm;
    private Tensor? _timbreEmbedW, _timbreEmbedB, _timbreSpecial, _timbreNorm;

    public AceStep15ConditionEncoder(AceStep15Config config)
    {
        _c = config;
        _lyricLayers = new AceStep15EncoderLayer[config.LyricEncoderLayers];
        _timbreLayers = new AceStep15EncoderLayer[config.TimbreEncoderLayers];
        for (int i = 0; i < _lyricLayers.Length; i++) _lyricLayers[i] = new AceStep15EncoderLayer(config);
        for (int i = 0; i < _timbreLayers.Length; i++) _timbreLayers[i] = new AceStep15EncoderLayer(config);
    }

    /// <summary>Loads the <c>encoder.*</c> keys of the main v1.5 safetensors (§5 of the research doc).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _textProjW = w["encoder.text_projector.weight"];
        _lyricEmbedW = w["encoder.lyric_encoder.embed_tokens.weight"];
        _lyricEmbedB = w["encoder.lyric_encoder.embed_tokens.bias"];
        _lyricNorm = EnsureF32(w["encoder.lyric_encoder.norm.weight"]);
        for (int i = 0; i < _lyricLayers.Length; i++)
            _lyricLayers[i].LoadWeights(w, $"encoder.lyric_encoder.layers.{i}");

        _timbreEmbedW = w["encoder.timbre_encoder.embed_tokens.weight"];
        _timbreEmbedB = w["encoder.timbre_encoder.embed_tokens.bias"];
        _timbreSpecial = EnsureF32(w["encoder.timbre_encoder.special_token"]);
        _timbreNorm = EnsureF32(w["encoder.timbre_encoder.norm.weight"]);
        for (int i = 0; i < _timbreLayers.Length; i++)
            _timbreLayers[i].LoadWeights(w, $"encoder.timbre_encoder.layers.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _textProjW, _lyricEmbedW, _lyricEmbedB, _lyricNorm,
            _timbreEmbedW, _timbreEmbedB, _timbreSpecial, _timbreNorm })
            if (t is not null) yield return t;
        foreach (AceStep15EncoderLayer l in _lyricLayers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
        foreach (AceStep15EncoderLayer l in _timbreLayers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Validates a hidden-states tensor as [T, hidden] or [1, T, hidden] (the latter is what
    /// <c>CfgHelper.SliceBatchElement</c> emits) and returns the sequence length T.</summary>
    private static int SeqLen(Tensor t, int hidden, string param, string label)
    {
        int rank = t.Shape.Rank;
        bool ok = (rank == 2 && (int)t.Shape[1] == hidden)
               || (rank == 3 && (int)t.Shape[0] == 1 && (int)t.Shape[2] == hidden);
        if (!ok)
            throw new ArgumentException($"{label} states must be [T, {hidden}] (or [1, T, {hidden}]); got {t.Shape}.", param);
        return (int)t.Shape[rank - 2];
    }

    /// <summary>Builds the packed condition sequence <c>[1, L_lyric + (timbre ? 1 : 0) + T_text, 2048]</c>.
    /// <paramref name="textHidden"/> and <paramref name="lyricHidden"/> are per-token Qwen3-Embedding-0.6B states
    /// <c>[T, 1024]</c>; <paramref name="timbreLatent"/> is a reference-audio Oobleck latent <c>[T_ref, 64]</c>
    /// (time-major), null for plain text-to-music.</summary>
    public Tensor EncodeConditions(IBackend backend, Tensor textHidden, Tensor? lyricHidden, Tensor? timbreLatent)
    {
        // Accept [T, H] or the [1, T, H] that CfgHelper.SliceBatchElement produces (the standard upstream
        // path: qwen.Encode → SliceBatchElement). backend.Linear runs on the flat buffer, so no reshape needed.
        int tText = SeqLen(textHidden, _c.TextHiddenDim, nameof(textHidden), "text");
        int dim = _c.HiddenSize;

        Tensor text = new Tensor(new TensorShape(1, tText, dim), DType.F32);
        backend.Linear(text, textHidden, _textProjW!, null);

        Tensor? lyric = null;
        if (lyricHidden is not null && (lyricHidden.Shape.Rank == 2 ? lyricHidden.Shape[0] : lyricHidden.Shape[1]) > 0)
        {
            int l = SeqLen(lyricHidden, _c.TextHiddenDim, nameof(lyricHidden), "lyric");
            Tensor embedded = new Tensor(new TensorShape(1, l, dim), DType.F32);
            backend.Linear(embedded, lyricHidden, _lyricEmbedW!, _lyricEmbedB);
            lyric = RunEncoder(backend, embedded, _lyricLayers, _lyricNorm!);
        }

        Tensor? timbre = null;
        if (timbreLatent is not null)
        {
            // CLS prepended, CLS row kept after the stack (the reference's "first position" aggregate) —
            // validation-gated vs the Python timbre_encoder until a real-checkpoint dump run.
            if (timbreLatent.Shape.Rank != 2 || (int)timbreLatent.Shape[1] != _c.TimbreHiddenDim)
                throw new ArgumentException($"timbre latent must be [T, {_c.TimbreHiddenDim}]; got {timbreLatent.Shape}.", nameof(timbreLatent));
            int tRef = (int)timbreLatent.Shape[0];
            Tensor embedded = new Tensor(new TensorShape(1, tRef + 1, dim), DType.F32);
            Buffer.MemoryCopy((float*)_timbreSpecial!.DataPointer, (float*)embedded.DataPointer, dim * 4, dim * 4);
            Tensor frames = new Tensor(new TensorShape(tRef, dim), DType.F32);
            backend.Linear(frames, timbreLatent, _timbreEmbedW!, _timbreEmbedB);
            Buffer.MemoryCopy((float*)frames.DataPointer, (float*)embedded.DataPointer + dim,
                (long)tRef * dim * 4, (long)tRef * dim * 4);
            frames.Dispose();
            Tensor encoded = RunEncoder(backend, embedded, _timbreLayers, _timbreNorm!);
            timbre = new Tensor(new TensorShape(1, 1, dim), DType.F32);
            Buffer.MemoryCopy((float*)encoded.DataPointer, (float*)timbre.DataPointer, dim * 4, dim * 4);
            encoded.Dispose();
        }

        int total = (lyric is null ? 0 : (int)lyric.Shape[1]) + (timbre is null ? 0 : 1) + tText;
        Tensor packed = new Tensor(new TensorShape(1, total, dim), DType.F32);
        float* pp = (float*)packed.DataPointer;
        long offset = 0;
        foreach (Tensor? part in new[] { lyric, timbre, text })
        {
            if (part is null) continue;
            long count = part.Shape[1] * dim;
            Buffer.MemoryCopy((float*)part.DataPointer, pp + offset, count * 4, count * 4);
            offset += count;
        }
        lyric?.Dispose();
        timbre?.Dispose();
        text.Dispose();
        return packed;
    }

    /// <summary>Runs a bidirectional Qwen3 stack (alternating sliding/full per <see cref="AceStep15Config.IsSlidingLayer"/>)
    /// plus the final RMSNorm; consumes <paramref name="x"/>.</summary>
    private Tensor RunEncoder(IBackend backend, Tensor x, AceStep15EncoderLayer[] layers, Tensor normWeight)
    {
        int s = (int)x.Shape[1];
        (float[] cos, float[] sin) = AceStep15Attention.BuildRopeTables(s, _c.HeadDim, _c.RopeTheta);
        Tensor? slidingMask = s > _c.SlidingWindow + 1 ? AceStep15Attention.BuildSlidingMask(s, _c.SlidingWindow) : null;
        Tensor cur = x;
        for (int i = 0; i < layers.Length; i++)
        {
            Tensor next = layers[i].Forward(backend, cur, cos, sin, _c.IsSlidingLayer(i) ? slidingMask : null);
            cur.Dispose();
            cur = next;
        }
        slidingMask?.Dispose();
        Tensor normed = new Tensor(cur.Shape, DType.F32);
        backend.RmsNorm(normed, cur, normWeight, _c.RmsNormEps);
        cur.Dispose();
        return normed;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
}
