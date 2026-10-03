using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5's text-to-speech GPT (<c>UnifiedVoice</c>): a standard biased HF <c>GPT2Model</c> body
/// (<see cref="GptBackbone"/>, keys translated by <see cref="IndexTtsGptKeyMap"/>) wrapped with IndexTTS's own
/// text/mel embedding tables, two SEPARATE learned position tables (text and mel each restart at 0 — see
/// <see cref="GptBackbone.Forward"/>'s <c>positionsApplied</c> doc), and IndexTTS's own second LayerNorm
/// (<c>final_norm</c>) applied on top of the GPT body's own internal <c>ln_f</c> — both are real, verified against
/// the checkpoint and the reference <c>GPT2InferenceModel.lm_head = Sequential(final_norm, mel_head)</c>.</summary>
/// <remarks>Two-pass generation, matching the reference <c>infer()</c> exactly: (1) autoregressively sample mel
/// codes with a KV-cache prefill + per-step decode, stopping at <c>stop_mel_token</c> or <c>max_mel_tokens</c>;
/// (2) the per-step <c>final_norm</c>'d hidden states collected during pass 1 ARE the "latent" sequence IndexTTS's
/// BigVGAN consumes directly — no separate non-causal re-forward pass is needed since this decoder captures each
/// step's post-final_norm hidden state as it goes, which is value-identical to recomputing it from the known
/// codes afterward (final_norm/ln_f are per-position, so a later full-sequence forward over the same tokens
/// reproduces the same hidden state at each position).</remarks>
internal sealed unsafe class IndexTtsT2sDecoder : IDisposable
{
    public const int StartMelToken = 8192;
    public const int StopMelToken = 8193;
    public const int NumMelCodes = 8194;
    public const int NumTextTokens = 12001;

    private readonly GptConfig _cfg;
    private readonly GptBackbone _gpt;
    private readonly int _maxMelTokens;
    private int _disposed;

    private Tensor? _textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB;

    public IndexTtsT2sDecoder(GptConfig cfg, int maxMelTokens)
    {
        _cfg = cfg;
        _gpt = new GptBackbone(cfg);
        _maxMelTokens = maxMelTokens;
    }

    /// <param name="w">Raw checkpoint tensors under the real <c>model.*</c> prefixes.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _textEmbed = WhisperOps.EnsureF32(w["model.text_embedding.weight"]);
        _textPos = WhisperOps.EnsureF32(w["model.text_pos_embedding.emb.weight"]);
        _melEmbed = WhisperOps.EnsureF32(w["model.mel_embedding.weight"]);
        _melPos = WhisperOps.EnsureF32(w["model.mel_pos_embedding.emb.weight"]);
        _finalNormW = WhisperOps.EnsureF32(w["model.final_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w["model.final_norm.bias"]);
        _melHeadW = WhisperOps.EnsureF32(w["model.mel_head.weight"]);
        _melHeadB = WhisperOps.EnsureF32(w["model.mel_head.bias"]);

        Dictionary<string, Tensor> translated = IndexTtsGptKeyMap.Translate(w, "model.gpt.h", "h", _cfg.NumLayers);
        _gpt.LoadWeights(translated, posKey: null, blockPrefix: "h", lnFGammaKey: "model.gpt.ln_f.weight", lnFBetaKey: "model.gpt.ln_f.bias");
    }

    /// <summary>Autoregressively generates mel-code latents conditioned on <paramref name="speechConditioning"/>
    /// (the Perceiver's 32-vector prefix, no position embedding) and <paramref name="textTokenIds"/> (embedded +
    /// text-positioned here). Returns <c>[1, melLen, hidden]</c> final_norm'd hidden states — IndexTTS's "latent",
    /// fed directly to BigVGAN (the sampled mel codes themselves are discarded once generation stops; see
    /// <see cref="Models.Codecs"/> and <c>docs/Research/INDEX_TTS_ARCHITECTURE.md</c> for why no codec decode is
    /// in this pipeline's inference path at all).</summary>
    public Tensor Generate(IBackend backend, Tensor speechConditioning, int[] textTokenIds, IndexTtsOptions options, Random rng)
    {
        if (_textEmbed is null) throw new InvalidOperationException("IndexTtsT2sDecoder weights not loaded.");
        int h = _cfg.Hidden;
        int condLen = (int)speechConditioning.Shape[1];
        int textLen = textTokenIds.Length;

        Tensor textEmb = EmbedWithPosition(textTokenIds, _textEmbed!, _textPos!, h);
        Tensor prefix = Concat(speechConditioning, textEmb, condLen, textLen, h);
        textEmb.Dispose();

        using IKvCache cache = _gpt.CreateCache();
        Tensor prefillOut = _gpt.Forward(backend, prefix, nonCausal: false, cache, positionsApplied: true);
        prefillOut.Dispose();
        prefix.Dispose();

        List<Tensor> latents = new(Math.Min(_maxMelTokens, 256));
        List<int> generated = new(latents.Capacity);
        int cap = Math.Min(options.MaxMelTokens ?? _maxMelTokens, _maxMelTokens);

        try
        {
            int prevToken = StartMelToken;
            for (int step = 0; step < cap; step++)
            {
                Tensor stepInput = EmbedOneWithPosition(prevToken, _melEmbed!, _melPos!, step, h);
                Tensor hiddenStep = _gpt.ForwardStep(backend, stepInput, cache, positionsApplied: true);
                stepInput.Dispose();

                Tensor normedStep = new(hiddenStep.Shape, DType.F32);
                backend.LayerNorm(normedStep, hiddenStep, _finalNormW!, _finalNormB!, 1e-5f);
                hiddenStep.Dispose();

                Tensor logits = WhisperOps.ProjectLinear(backend, normedStep, _melHeadW!, _melHeadB, 1, 1, h, NumMelCodes);
                int nextToken = SampleNextToken(logits, generated, options, rng);
                logits.Dispose();

                if (nextToken == StopMelToken)
                {
                    normedStep.Dispose();
                    break;
                }
                latents.Add(normedStep);
                generated.Add(nextToken);
                prevToken = nextToken;
            }

            if (latents.Count == 0)
                throw new InvalidOperationException("IndexTTS generated zero mel frames (immediate stop token).");

            Tensor latent = new(new TensorShape(1, latents.Count, h), DType.F32);
            float* lp = (float*)latent.DataPointer;
            for (int i = 0; i < latents.Count; i++)
            {
                float* src = (float*)latents[i].DataPointer;
                for (int c = 0; c < h; c++) lp[(long)i * h + c] = src[c];
            }
            return latent;
        }
        finally
        {
            foreach (Tensor t in latents) t.Dispose();
        }
    }

    /// <summary>Top-k temperature sampling with a multiplicative repetition penalty (CTRL-style: positive logits
    /// divided by the penalty, negative ones multiplied) over already-generated codes — <see cref="LogitSampling"/>
    /// has no built-in penalty, and IndexTTS's reference default (repetition_penalty≈10 in the upstream CLI) makes
    /// this matter in practice for avoiding stuck/looping codes.</summary>
    private static int SampleNextToken(Tensor logits, List<int> generated, IndexTtsOptions options, Random rng)
    {
        float* lp = (float*)logits.DataPointer;
        int n = (int)logits.ElementCount;
        Span<float> span = new(lp, n);
        if (options.RepetitionPenalty != 1f && generated.Count > 0)
        {
            float penalty = options.RepetitionPenalty;
            foreach (int id in generated)
            {
                float v = span[id];
                span[id] = v > 0 ? v / penalty : v * penalty;
            }
        }
        return options.Temperature <= 0f
            ? LogitSampling.ArgMax(span)
            : LogitSampling.SampleTopK(span, options.Temperature, options.TopK, rng);
    }

    /// <summary>Embeds <paramref name="ids"/> and adds the matching learned position (index == sequence position,
    /// restarting at 0) — host loop, since this runs once per reference/prompt, not per generated token.</summary>
    private static Tensor EmbedWithPosition(int[] ids, Tensor table, Tensor posTable, int hidden)
    {
        Tensor outT = new(new TensorShape(1, ids.Length, hidden), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* tp = (float*)table.DataPointer;
        float* pp = (float*)posTable.DataPointer;
        for (int i = 0; i < ids.Length; i++)
        {
            float* tok = tp + (long)ids[i] * hidden;
            float* pos = pp + (long)i * hidden;
            float* dst = op + (long)i * hidden;
            for (int c = 0; c < hidden; c++) dst[c] = tok[c] + pos[c];
        }
        return outT;
    }

    private static Tensor EmbedOneWithPosition(int id, Tensor table, Tensor posTable, int position, int hidden)
    {
        Tensor outT = new(new TensorShape(1, 1, hidden), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* tok = (float*)table.DataPointer + (long)id * hidden;
        float* pos = (float*)posTable.DataPointer + (long)position * hidden;
        for (int c = 0; c < hidden; c++) op[c] = tok[c] + pos[c];
        return outT;
    }

    private static Tensor Concat(Tensor a, Tensor b, int aLen, int bLen, int hidden)
    {
        Tensor outT = new(new TensorShape(1, aLen + bLen, hidden), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        long aCount = (long)aLen * hidden;
        for (long i = 0; i < aCount; i++) op[i] = ap[i];
        long bCount = (long)bLen * hidden;
        for (long i = 0; i < bCount; i++) op[aCount + i] = bp[i];
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor t in _gpt.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _gpt.Dispose();
        GC.SuppressFinalize(this);
    }
}
