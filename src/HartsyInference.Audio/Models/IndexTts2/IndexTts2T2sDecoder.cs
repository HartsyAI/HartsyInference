using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's text-to-speech GPT (<c>UnifiedVoice</c>, real <c>spk_cond_mode="campplus"</c> path —
/// IndexTTS-2.5's real deployed mode, confirmed from the checkpoint's own keys: <c>spk_emb_proj</c> +
/// <c>lang_embedding</c> present, <c>conditioning_encoder</c>/<c>perceiver_encoder</c>/<c>speed_emb</c> absent).
/// IndexTTS-2.0's real checkpoint uses the OTHER mode instead (<c>condition_type: conformer_perceiver</c>, same
/// shape as <see cref="IndexTtsSpeakerEncoder"/>) — not yet implemented here; this class is 2.5-first per the
/// project's stated priority.
/// <para>Unlike IndexTTS-1.5's GPT, this one's output IS the final discrete artifact (semantic-codec codes fed
/// to <see cref="HartsyInference.Audio.Models.Codecs.VocosFactorizedCodec.Decode"/>), not a continuous latent
/// handed to a vocoder — so <see cref="Generate"/> is genuinely single-pass (sample, stop at
/// <see cref="StopMelToken"/>, return the raw code sequence), unlike IndexTTS-1.5's two-pass
/// <c>IndexTtsT2sDecoder.Generate</c>. Confirmed from the real <c>infer_v2_5.py</c>: <c>codes, _ =
/// gpt.inference_speech(...)</c> never re-forwards for hidden states.</para></summary>
internal sealed unsafe class IndexTts2T2sDecoder : IDisposable
{
    public const int StartMelToken = 8192;
    public const int StopMelToken = 8193;
    public const int NumMelCodes = 8194;
    public const int StartTextToken = 0;
    public const int StopTextToken = 1;

    private readonly GptConfig _cfg;
    private readonly int _numTextTokens, _maxMelTokens, _emoPerceiverDim;
    private readonly GptBackbone _gpt;
    private readonly IndexTtsConformerEncoder _emoConformer;
    private readonly IndexTtsPerceiver _emoPerceiver;
    private int _disposed;

    private Tensor? _textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB;
    private Tensor? _spkEmbProjW, _spkEmbProjB, _langEmbed;
    private Tensor? _emovecLayerW, _emovecLayerB, _emoLayerW, _emoLayerB;
    private Tensor[]? _ownedGptWeights;

    /// <param name="emoConditionCfg">Defaults to the real <see cref="IndexTtsConformerConfig.IndexTts2EmoCondition"/>
    /// preset; overridable so tests can exercise this class with a tiny Conformer instead of the real 1024-in/512-out/
    /// 4-block one.</param>
    /// <param name="emoPerceiverDim">The emotion Perceiver's own working width — a real, fixed value (1024) on the
    /// upstream <c>PerceiverResampler(1024, dim_context=emo_condition_module.output_size, ...)</c> call,
    /// independent of <paramref name="emoConditionCfg"/>'s own dims; overridable for the same reason.</param>
    public IndexTts2T2sDecoder(GptConfig cfg, int numTextTokens, int maxMelTokens,
        IndexTtsConformerConfig? emoConditionCfg = null, int emoPerceiverDim = 1024)
    {
        _cfg = cfg;
        _numTextTokens = numTextTokens;
        _maxMelTokens = maxMelTokens;
        _emoPerceiverDim = emoPerceiverDim;
        _gpt = new GptBackbone(cfg);
        IndexTtsConformerConfig emoCfg = emoConditionCfg ?? IndexTtsConformerConfig.IndexTts2EmoCondition;
        _emoConformer = new IndexTtsConformerEncoder(emoCfg);
        _emoPerceiver = new IndexTtsPerceiver(dim: emoPerceiverDim, dimContext: emoCfg.OutputSize, depth: 2, numLatents: 1, heads: 4);
    }

    /// <param name="w">Raw checkpoint tensors under their real top-level names (IndexTTS-2's <c>gpt.pth</c> has
    /// no <c>model.</c> prefix, unlike IndexTTS-1.5's).</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _textEmbed = WhisperOps.EnsureF32(w["text_embedding.weight"]);
        _textPos = WhisperOps.EnsureF32(w["text_pos_embedding.emb.weight"]);
        _melEmbed = WhisperOps.EnsureF32(w["mel_embedding.weight"]);
        _melPos = WhisperOps.EnsureF32(w["mel_pos_embedding.emb.weight"]);
        _finalNormW = WhisperOps.EnsureF32(w["final_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w["final_norm.bias"]);
        _melHeadW = WhisperOps.EnsureF32(w["mel_head.weight"]);
        _melHeadB = WhisperOps.EnsureF32(w["mel_head.bias"]);

        _spkEmbProjW = WhisperOps.EnsureF32(w["spk_emb_proj.weight"]);
        _spkEmbProjB = WhisperOps.EnsureF32(w["spk_emb_proj.bias"]);
        _langEmbed = WhisperOps.EnsureF32(w["lang_embedding.weight"]);

        _emovecLayerW = WhisperOps.EnsureF32(w["emovec_layer.weight"]);
        _emovecLayerB = WhisperOps.EnsureF32(w["emovec_layer.bias"]);
        _emoLayerW = WhisperOps.EnsureF32(w["emo_layer.weight"]);
        _emoLayerB = WhisperOps.EnsureF32(w["emo_layer.bias"]);
        _emoConformer.LoadWeights(w, "emo_conditioning_encoder");
        _emoPerceiver.LoadWeights(w, "emo_perceiver_encoder");

        (Dictionary<string, Tensor> translated, Tensor[] owned) = IndexTtsGptKeyMap.Translate(w, "gpt.h", "h", _cfg.NumLayers);
        _ownedGptWeights = owned;
        _gpt.LoadWeights(translated, posKey: null, blockPrefix: "h", lnFGammaKey: "gpt.ln_f.weight", lnFBetaKey: "gpt.ln_f.bias");
    }

    /// <summary>Real <c>spk_emb_proj(campplusEmbedding)</c>: projects the CAM++ style embedding <c>[1, 192]</c>
    /// (the SAME vector already computed for S2Mel's style conditioning — one CAM++ forward pass serves both)
    /// up to the GPT's own hidden size, returning <c>[1, 1, hidden]</c>.</summary>
    public Tensor ComputeSpeakerConditioning(IBackend backend, Tensor campplusEmbedding)
    {
        ThrowIfDisposed();
        return WhisperOps.ProjectLinear(backend, campplusEmbedding, _spkEmbProjW!, _spkEmbProjB, 1, 1, 192, _cfg.Hidden);
    }

    /// <summary>Real <c>get_emovec</c>: <c>emo_layer(emovec_layer(get_emo_conditioning(refFeature)))</c> — the
    /// emotion Conformer+Perceiver pooling a w2v-bert reference feature <c>[1, T, 1024]</c> down to one vector,
    /// then two Linear layers up to the GPT's hidden size. Called on either the main speaker's own reference audio
    /// (the "base" emovec) or a separate emotion-reference clip — see <see cref="MergeEmoVec"/>.</summary>
    public Tensor ComputeEmoVec(IBackend backend, Tensor refFeature, int t)
    {
        ThrowIfDisposed();
        Tensor conformerOut = _emoConformer.Forward(backend, refFeature, t);
        int tOut = (int)conformerOut.Shape[1];
        Tensor pooled = _emoPerceiver.Forward(backend, conformerOut, tOut);  // [1, 1, emoPerceiverDim]
        conformerOut.Dispose();
        Tensor emovecSyn = WhisperOps.ProjectLinear(backend, pooled, _emovecLayerW!, _emovecLayerB, 1, 1, _emoPerceiverDim, _cfg.Hidden);
        pooled.Dispose();
        Tensor emoVec = WhisperOps.ProjectLinear(backend, emovecSyn, _emoLayerW!, _emoLayerB, 1, 1, _cfg.Hidden, _cfg.Hidden);
        emovecSyn.Dispose();
        return emoVec;
    }

    /// <summary>Real <c>merge_emovec</c>: <c>base + alpha * (emo - base)</c> — <c>alpha=0</c> keeps the speaker's
    /// own natural emotion from their main reference clip, <c>alpha=1</c> fully adopts a separate emotion
    /// reference's. Both inputs are <see cref="ComputeEmoVec"/> outputs <c>[1, hidden]</c> or <c>[1, 1, hidden]</c>.</summary>
    public static Tensor MergeEmoVec(Tensor baseVec, Tensor emoVec, float alpha)
    {
        Tensor result = new(baseVec.Shape, DType.F32);
        float* bp = (float*)baseVec.DataPointer;
        float* ep = (float*)emoVec.DataPointer;
        float* rp = (float*)result.DataPointer;
        long n = baseVec.ElementCount;
        for (long i = 0; i < n; i++) rp[i] = bp[i] + alpha * (ep[i] - bp[i]);
        return result;
    }

    /// <summary>Single-pass autoregressive generation of semantic-codec codes. Real conditioning assembly
    /// (<c>inference_speech</c>, campplus mode): <c>conds = cat([speakerConditioning + emoVec, zeros(1,2,hidden)],
    /// dim=1)</c> — one speaker+emotion slot plus two always-zero pad slots (the real <c>duration_emb</c>
    /// placeholder for campplus mode is literally these two zero slots, not an embedding lookup at all — the
    /// non-campplus mode's real <c>speed_emb</c> is a separate, confirmed-near-zero/untrained embedding this
    /// class does not need). Text gets <c>text_embedding[id] + text_pos_embedding[i]</c>, plus
    /// <c>lang_embedding[langId]</c> broadcast across every text position when <paramref name="langId"/> is given
    /// (real <c>prepare_gpt_inputs</c>: only added for campplus mode, which this class always is). No position
    /// embedding on the conditioning slots, matching the real source.</summary>
    public int[] Generate(IBackend backend, Tensor speakerConditioning, Tensor emoVec, int[] textTokenIds,
        int? langId, IndexTtsOptions options, ref uint rngState)
    {
        ThrowIfDisposed();
        int h = _cfg.Hidden;

        Tensor conds = BuildCampplusConditioning(speakerConditioning, emoVec, h);
        int condLen = 3;

        int[] wrappedTextIds = new int[textTokenIds.Length + 2];
        wrappedTextIds[0] = StartTextToken;
        Array.Copy(textTokenIds, 0, wrappedTextIds, 1, textTokenIds.Length);
        wrappedTextIds[^1] = StopTextToken;
        int textLen = wrappedTextIds.Length;

        int cap = Math.Min(options.MaxMelTokens ?? _maxMelTokens, _maxMelTokens);
        cap = Math.Min(cap, _cfg.BlockSize - condLen - textLen - 1);
        if (cap <= 0)
        {
            conds.Dispose();
            throw new InvalidOperationException(
                $"Conditioning ({condLen}) + text ({textLen}) tokens leave no room for mel generation within block size {_cfg.BlockSize}.");
        }

        List<int> generated = new(Math.Min(cap, 256));
        HashSet<int> seenForPenalty = [StartMelToken];
        using (Tensor textEmb = EmbedTextWithPositionAndLang(wrappedTextIds, langId, h))
        using (Tensor prefix = Concat(conds, textEmb, condLen, textLen, h))
        using (IKvCache cache = _gpt.CreateCache())
        {
            conds.Dispose();
            _gpt.Forward(backend, prefix, nonCausal: false, cache, positionsApplied: true).Dispose();

            int prevToken = StartMelToken;
            for (int step = 0; step < cap; step++)
            {
                using Tensor stepInput = EmbedOneWithPosition(prevToken, _melEmbed!, _melPos!, step, h);
                using Tensor hiddenStep = _gpt.ForwardStep(backend, stepInput, cache, positionsApplied: true);
                using Tensor normedStep = new(hiddenStep.Shape, DType.F32);
                backend.LayerNorm(normedStep, hiddenStep, _finalNormW!, _finalNormB!, 1e-5f);

                using Tensor logits = WhisperOps.ProjectLinear(backend, normedStep, _melHeadW!, _melHeadB, 1, 1, h, NumMelCodes);
                int nextToken = SampleNextToken(logits, seenForPenalty, options, ref rngState);
                if (nextToken == StopMelToken) break;
                generated.Add(nextToken);
                seenForPenalty.Add(nextToken);
                prevToken = nextToken;
            }
        }

        if (generated.Count == 0)
            throw new InvalidOperationException("IndexTTS-2 generated zero semantic codes (immediate stop token).");
        return [.. RemoveLongSilence(generated)];
    }

    /// <summary><c>cat([speakerConditioning + emoVec.unsqueeze(1), zeros(1,2,hidden)], dim=1)</c> → <c>[1, 3, hidden]</c>.</summary>
    private static Tensor BuildCampplusConditioning(Tensor speakerConditioning, Tensor emoVec, int h)
    {
        Tensor conds = new(new TensorShape(1, 3, h), DType.F32);
        float* cp = (float*)conds.DataPointer;
        float* sp = (float*)speakerConditioning.DataPointer;
        float* ep = (float*)emoVec.DataPointer;
        for (int c = 0; c < h; c++) cp[c] = sp[c] + ep[c];
        for (int c = h; c < 3 * h; c++) cp[c] = 0f;
        return conds;
    }

    private Tensor EmbedTextWithPositionAndLang(int[] ids, int? langId, int hidden)
    {
        Tensor outT = new(new TensorShape(1, ids.Length, hidden), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* tp = (float*)_textEmbed!.DataPointer;
        float* pp = (float*)_textPos!.DataPointer;
        float* langRow = langId is int lid ? (float*)_langEmbed!.DataPointer + (long)lid * hidden : null;
        for (int i = 0; i < ids.Length; i++)
        {
            float* tok = tp + (long)ids[i] * hidden;
            float* pos = pp + (long)i * hidden;
            float* dst = op + (long)i * hidden;
            for (int c = 0; c < hidden; c++) dst[c] = tok[c] + pos[c] + (langRow is not null ? langRow[c] : 0f);
        }
        return outT;
    }

    /// <summary>Temperature/top-k/top-p sampling with a multiplicative repetition penalty — identical scheme to
    /// IndexTTS-1.5's <c>IndexTtsT2sDecoder.SampleNextToken</c> (same <see cref="NucleusSampler"/>, same CTRL-style
    /// penalty). Beam search (the reference's real default) is not implemented, same documented gap as Phase 1.</summary>
    private static int SampleNextToken(Tensor logits, HashSet<int> seen, IndexTtsOptions options, ref uint rngState)
    {
        float* lp = (float*)logits.DataPointer;
        int n = (int)logits.ElementCount;
        Span<float> span = new(lp, n);
        if (options.RepetitionPenalty != 1f)
        {
            float penalty = options.RepetitionPenalty;
            foreach (int id in seen)
            {
                float v = span[id];
                span[id] = v > 0 ? v / penalty : v * penalty;
            }
        }
        return options.Temperature <= 0f
            ? LogitSampling.ArgMax(span)
            : NucleusSampler.Draw(span, n, options.Temperature, options.TopK, options.TopP, ref rngState);
    }

    /// <summary>Same real <c>remove_long_silence</c> cap as IndexTTS-1.5's decoder — see
    /// <c>IndexTtsT2sDecoder.RemoveLongSilence</c> for the full rationale; identical logic, duplicated rather than
    /// shared because both are tiny (~10 lines) private helpers on otherwise-unrelated classes.</summary>
    private static List<int> RemoveLongSilence(List<int> codes, int silentToken = 52, int maxTotal = 30, int keepRun = 10)
    {
        int total = 0;
        foreach (int c in codes) if (c == silentToken) total++;
        if (total <= maxTotal) return codes;

        List<int> trimmed = new(codes.Count);
        int run = 0;
        foreach (int c in codes)
        {
            if (c != silentToken) { trimmed.Add(c); run = 0; }
            else if (run < keepRun) { trimmed.Add(c); run++; }
        }
        return trimmed;
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
        Tensor?[] core =
        [
            _textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB,
            _spkEmbProjW, _spkEmbProjB, _langEmbed, _emovecLayerW, _emovecLayerB, _emoLayerW, _emoLayerB,
        ];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor t in _emoConformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _emoPerceiver.EnumerateWeights()) yield return t;
        foreach (Tensor t in _gpt.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTts2T2sDecoder));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _gpt.Dispose();
        _emoConformer.Dispose();
        _emoPerceiver.Dispose();
        if (_ownedGptWeights is not null) foreach (Tensor t in _ownedGptWeights) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
