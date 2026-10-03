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
/// <remarks>Two-pass generation, matching the reference <c>infer()</c> exactly — see <see cref="Generate"/>'s own
/// remarks for why both passes are required (the per-step hidden states from the sampling pass are NOT the ones
/// BigVGAN needs; a second non-cached forward over the known sequence is).</remarks>
internal sealed unsafe class IndexTtsT2sDecoder : IDisposable
{
    public const int StartMelToken = 8192;
    public const int StopMelToken = 8193;
    public const int NumMelCodes = 8194;
    public const int NumTextTokens = 12001;
    public const int StartTextToken = 0;
    public const int StopTextToken = 1;

    private readonly GptConfig _cfg;
    private readonly GptBackbone _gpt;
    private readonly int _maxMelTokens;
    private int _disposed;

    private Tensor? _textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB;
    private Tensor[]? _ownedGptWeights;

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

        (Dictionary<string, Tensor> translated, Tensor[] owned) = IndexTtsGptKeyMap.Translate(w, "model.gpt.h", "h", _cfg.NumLayers);
        _ownedGptWeights = owned;
        _gpt.LoadWeights(translated, posKey: null, blockPrefix: "h", lnFGammaKey: "model.gpt.ln_f.weight", lnFBetaKey: "model.gpt.ln_f.bias");
    }

    /// <summary>Generates mel-code latents conditioned on <paramref name="speechConditioning"/> (the Perceiver's
    /// 32-vector prefix, no position embedding) and <paramref name="textTokenIds"/> (wrapped with
    /// <see cref="StartTextToken"/>/<see cref="StopTextToken"/>, embedded, and text-positioned — the real
    /// <c>prepare_gpt_inputs</c> pads every text sequence with these sentinels before embedding, confirmed against
    /// the reference source). Returns <c>[1, melLen, hidden]</c> final_norm'd hidden states — IndexTTS's "latent",
    /// fed directly to BigVGAN.</summary>
    /// <remarks>Genuinely two passes, matching the reference exactly (an earlier version of this method tried to
    /// collect latents inline during the one sampling pass — wrong, because the hidden state produced by feeding
    /// token i-1 as input is NOT the same as the hidden state at token i's own position: causal attention means
    /// "the state that predicts token i" and "the state produced by token i as input" are off by one position,
    /// and the inline version was collecting the former under the latter's index while also never computing a
    /// state for the final sampled code at all). Pass 1 samples the code sequence only (no latent collection:
    /// those hidden states are the wrong ones). Pass 2 re-embeds the now-known full sequence
    /// <c>[conditioning; text; StartMelToken; sampled codes]</c> and runs ONE non-cached causal forward over it —
    /// matching the reference's <c>UnifiedVoice.forward(..., return_latent=True)</c>, which pads its mel sequence
    /// to <c>[start_mel, codes..., stop_mel, stop_mel]</c> and then strips exactly the trailing two positions
    /// (<c>mel_logits[:, :-2]</c>), netting <c>[start_mel's own hidden state, then one state per generated
    /// code]</c> — <c>N+1</c> latent frames for <c>N</c> generated codes, NOT <c>N</c>; BigVGAN is conditioned on
    /// that extra leading frame too, confirmed by reading the real <c>index-tts</c> GitHub source directly (an
    /// earlier version of this method omitted the start token and returned only <c>N</c> frames, which would both
    /// shorten the waveform by one frame and shift every mel position embedding by one relative to pass 1's own
    /// usage). Then slices out the mel segment's hidden states and applies <c>final_norm</c> (LayerNorm is
    /// per-position, so normalizing the slice equals normalizing the whole sequence first).</remarks>
    public Tensor Generate(IBackend backend, Tensor speechConditioning, int[] textTokenIds, IndexTtsOptions options, Random rng)
    {
        if (_textEmbed is null) throw new InvalidOperationException("IndexTtsT2sDecoder weights not loaded.");
        int h = _cfg.Hidden;
        int condLen = (int)speechConditioning.Shape[1];

        int[] wrappedTextIds = new int[textTokenIds.Length + 2];
        wrappedTextIds[0] = StartTextToken;
        Array.Copy(textTokenIds, 0, wrappedTextIds, 1, textTokenIds.Length);
        wrappedTextIds[^1] = StopTextToken;
        int textLen = wrappedTextIds.Length;

        int cap = Math.Min(options.MaxMelTokens ?? _maxMelTokens, _maxMelTokens);
        // Leave room in the KV cache for conditioning + text before the decode loop starts consuming it, so a
        // long prompt can't run ForwardStep past the cache's capacity and throw mid-generation. The extra -1
        // reserves a slot for pass 2's re-embedding, whose mel segment is N+1 long (StartMelToken plus every
        // generated code) — one longer than pass 1's own cache ever holds, since the AR loop never feeds the
        // last sampled code back as an input.
        cap = Math.Min(cap, _cfg.BlockSize - condLen - textLen - 1);
        if (cap <= 0)
        {
            throw new InvalidOperationException(
                $"Conditioning ({condLen}) + text ({textLen}) tokens leave no room for mel generation within block size {_cfg.BlockSize}.");
        }

        List<int> generated = new(Math.Min(cap, 256));
        using (Tensor textEmb = EmbedWithPosition(wrappedTextIds, _textEmbed!, _textPos!, h))
        using (Tensor prefix = Concat(speechConditioning, textEmb, condLen, textLen, h))
        using (IKvCache cache = _gpt.CreateCache())
        {
            _gpt.Forward(backend, prefix, nonCausal: false, cache, positionsApplied: true).Dispose();

            int prevToken = StartMelToken;
            for (int step = 0; step < cap; step++)
            {
                using Tensor stepInput = EmbedOneWithPosition(prevToken, _melEmbed!, _melPos!, step, h);
                using Tensor hiddenStep = _gpt.ForwardStep(backend, stepInput, cache, positionsApplied: true);
                using Tensor normedStep = new(hiddenStep.Shape, DType.F32);
                backend.LayerNorm(normedStep, hiddenStep, _finalNormW!, _finalNormB!, 1e-5f);

                using Tensor logits = WhisperOps.ProjectLinear(backend, normedStep, _melHeadW!, _melHeadB, 1, 1, h, NumMelCodes);
                int nextToken = SampleNextToken(logits, generated, options, rng);
                if (nextToken == StopMelToken) break;
                generated.Add(nextToken);
                prevToken = nextToken;
            }
        }

        if (generated.Count == 0)
            throw new InvalidOperationException("IndexTTS generated zero mel frames (immediate stop token).");

        int[] melWithStart = new int[generated.Count + 1];
        melWithStart[0] = StartMelToken;
        generated.CopyTo(melWithStart, 1);
        int melSegLen = melWithStart.Length; // N+1: StartMelToken's own state plus one per generated code.

        using (Tensor textEmb2 = EmbedWithPosition(wrappedTextIds, _textEmbed!, _textPos!, h))
        using (Tensor condText = Concat(speechConditioning, textEmb2, condLen, textLen, h))
        using (Tensor melEmb = EmbedWithPosition(melWithStart, _melEmbed!, _melPos!, h))
        using (Tensor fullSeq = Concat(condText, melEmb, condLen + textLen, melSegLen, h))
        using (Tensor fullOut = _gpt.Forward(backend, fullSeq, nonCausal: false, cache: null, positionsApplied: true))
        using (Tensor melSlice = SliceSequence(fullOut, condLen + textLen, melSegLen, h))
        {
            Tensor latent = new(melSlice.Shape, DType.F32);
            backend.LayerNorm(latent, melSlice, _finalNormW!, _finalNormB!, 1e-5f);
            return latent;
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

    /// <summary>Slices <paramref name="seq"/> <c>[1, totalLen, hidden]</c> to <c>[1, length, hidden]</c> starting at <paramref name="offset"/>.</summary>
    private static Tensor SliceSequence(Tensor seq, int offset, int length, int hidden)
    {
        Tensor outT = new(new TensorShape(1, length, hidden), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* sp = (float*)seq.DataPointer + (long)offset * hidden;
        long count = (long)length * hidden;
        for (long i = 0; i < count; i++) op[i] = sp[i];
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
        // GptBlock.Dispose() is a no-op by design (its weights are normally borrowed references the checkpoint
        // loader owns) — but IndexTtsGptKeyMap.Translate allocates genuinely new transposed tensors nothing else
        // references, so they must be disposed here instead.
        if (_ownedGptWeights is not null) foreach (Tensor t in _ownedGptWeights) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
