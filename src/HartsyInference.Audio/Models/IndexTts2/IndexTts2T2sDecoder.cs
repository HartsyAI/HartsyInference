using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>How the GPT's speaker conditioning is produced — the real <c>UnifiedVoice.spk_cond_mode</c>, decided by
/// which tensors the loaded checkpoint carries.</summary>
internal enum IndexTts2SpeakerConditioning
{
    /// <summary>IndexTTS-2.5: <c>spk_emb_proj</c> over the CAM++ style vector (one <c>[1,1,hidden]</c> slot) plus
    /// <c>lang_embedding</c>; <c>conditioning_encoder</c>/<c>perceiver_encoder</c>/<c>speed_emb</c> absent.</summary>
    Campplus,

    /// <summary>IndexTTS-2.0 (<c>condition_type: conformer_perceiver</c>): a Conformer+Perceiver over the w2v-bert
    /// feature (<see cref="IndexTtsSpeakerEncoder"/> unchanged) giving 32 latents, plus two real
    /// <c>speed_emb</c> lookup slots; no <c>lang_embedding</c>.</summary>
    ConformerPerceiver,
}

/// <summary>IndexTTS-2's text-to-speech GPT (<c>UnifiedVoice</c>), covering both released checkpoints:
/// IndexTTS-2.5 (<see cref="IndexTts2SpeakerConditioning.Campplus"/>) and IndexTTS-2.0
/// (<see cref="IndexTts2SpeakerConditioning.ConformerPerceiver"/>). <see cref="LoadWeights"/> picks the mode from
/// the checkpoint's own keys, so the pipeline never needs a second decoder type.
/// <para>Generation is single-pass sampling of semantic-codec codes (<see cref="Generate"/>). 2.5 hands those codes
/// straight to its bundled codec's <c>Decode</c>. 2.0 additionally re-forwards the known codes once
/// (<see cref="ComputeSecondPassLatent"/>, the real <c>self.gpt(...)</c> call in <c>infer_v2.py</c>) and feeds the
/// result through <c>s2mel.gpt_layer</c> (<see cref="LoadGptLayer"/>), summed with the codebook embedding.</para></summary>
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
    private readonly IndexTtsConformerConfig _speakerConditionCfg;
    private IndexTtsSpeakerEncoder? _speakerEncoder;
    private int _disposed;

    private Tensor? _textEmbed, _textPos, _melEmbed, _melPos, _finalNormW, _finalNormB, _melHeadW, _melHeadB;
    private Tensor? _spkEmbProjW, _spkEmbProjB, _langEmbed, _speedEmbed;
    private Tensor?[]? _gptLayerW, _gptLayerB;
    private Tensor? _emovecLayerW, _emovecLayerB, _emoLayerW, _emoLayerB;
    private Tensor[]? _ownedGptWeights;

    /// <param name="emoConditionCfg">Defaults to the real <see cref="IndexTtsConformerConfig.IndexTts2EmoCondition"/>
    /// preset; overridable so tests can exercise this class with a tiny Conformer instead of the real 1024-in/512-out/
    /// 4-block one.</param>
    /// <param name="emoPerceiverDim">The emotion Perceiver's own working width — a real, fixed value (1024) on the
    /// upstream <c>PerceiverResampler(1024, dim_context=emo_condition_module.output_size, ...)</c> call,
    /// independent of <paramref name="emoConditionCfg"/>'s own dims; overridable for the same reason.</param>
    /// <param name="speakerConditionCfg">The 2.0-mode speaker Conformer preset
    /// (<see cref="IndexTtsConformerConfig.IndexTts2Condition"/> by default); only used when the loaded checkpoint
    /// is <see cref="IndexTts2SpeakerConditioning.ConformerPerceiver"/>.</param>
    public IndexTts2T2sDecoder(GptConfig cfg, int numTextTokens, int maxMelTokens,
        IndexTtsConformerConfig? emoConditionCfg = null, int emoPerceiverDim = 1024,
        IndexTtsConformerConfig? speakerConditionCfg = null)
    {
        _speakerConditionCfg = speakerConditionCfg ?? IndexTtsConformerConfig.IndexTts2Condition;
        _cfg = cfg;
        _numTextTokens = numTextTokens;
        _maxMelTokens = maxMelTokens;
        _emoPerceiverDim = emoPerceiverDim;
        _gpt = new GptBackbone(cfg);
        IndexTtsConformerConfig emoCfg = emoConditionCfg ?? IndexTtsConformerConfig.IndexTts2EmoCondition;
        _emoConformer = new IndexTtsConformerEncoder(emoCfg);
        _emoPerceiver = new IndexTtsPerceiver(dim: emoPerceiverDim, dimContext: emoCfg.OutputSize, depth: 2, numLatents: 1, heads: 4);
    }

    /// <summary>Which speaker-conditioning path <see cref="LoadWeights"/> found in the checkpoint.</summary>
    public IndexTts2SpeakerConditioning Mode { get; private set; } = IndexTts2SpeakerConditioning.Campplus;

    /// <param name="w">Raw checkpoint tensors under their real top-level names (IndexTTS-2's <c>gpt.pth</c> has
    /// no <c>model.</c> prefix, unlike IndexTTS-1.5's).</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Mode = w.ContainsKey("speed_emb.weight") ? IndexTts2SpeakerConditioning.ConformerPerceiver : IndexTts2SpeakerConditioning.Campplus;
        _textEmbed = WhisperOps.EnsureF32(w["text_embedding.weight"]);
        _textPos = WhisperOps.EnsureF32(w["text_pos_embedding.emb.weight"]);
        _melEmbed = WhisperOps.EnsureF32(w["mel_embedding.weight"]);
        _melPos = WhisperOps.EnsureF32(w["mel_pos_embedding.emb.weight"]);
        _finalNormW = WhisperOps.EnsureF32(w["final_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w["final_norm.bias"]);
        _melHeadW = WhisperOps.EnsureF32(w["mel_head.weight"]);
        _melHeadB = WhisperOps.EnsureF32(w["mel_head.bias"]);

        if (Mode == IndexTts2SpeakerConditioning.Campplus)
        {
            _spkEmbProjW = WhisperOps.EnsureF32(w["spk_emb_proj.weight"]);
            _spkEmbProjB = WhisperOps.EnsureF32(w["spk_emb_proj.bias"]);
            _langEmbed = WhisperOps.EnsureF32(w["lang_embedding.weight"]);
        }
        else
        {
            _speedEmbed = WhisperOps.EnsureF32(w["speed_emb.weight"]);
            _speakerEncoder = new IndexTtsSpeakerEncoder(_speakerConditionCfg, _cfg.Hidden);
            _speakerEncoder.LoadWeights(w, "conditioning_encoder", "perceiver_encoder");
        }

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
    /// up to the GPT's own hidden size, returning <c>[1, 1, hidden]</c>. <see cref="IndexTts2SpeakerConditioning.Campplus"/> only.</summary>
    public Tensor ComputeSpeakerConditioning(IBackend backend, Tensor campplusEmbedding)
    {
        ThrowIfDisposed();
        RequireMode(IndexTts2SpeakerConditioning.Campplus);
        return WhisperOps.ProjectLinear(backend, campplusEmbedding, _spkEmbProjW!, _spkEmbProjB, 1, 1, 192, _cfg.Hidden);
    }

    /// <summary>Real <c>get_conditioning</c> for <c>condition_type: conformer_perceiver</c>: the w2v-bert reference
    /// feature <c>[1, T, 1024]</c> through the speaker Conformer and the 32-latent Perceiver, returning
    /// <c>[1, 32, hidden]</c>. <see cref="IndexTts2SpeakerConditioning.ConformerPerceiver"/> only.</summary>
    public Tensor ComputeSpeakerConditioningConformerPerceiver(IBackend backend, Tensor w2vBertFeature, int t)
    {
        ThrowIfDisposed();
        RequireMode(IndexTts2SpeakerConditioning.ConformerPerceiver);
        return _speakerEncoder!.Forward(backend, w2vBertFeature, t);
    }

    /// <summary>Loads <c>s2mel.gpt_layer</c> — the real <c>nn.Sequential(Linear(1280,256), Linear(256,128),
    /// Linear(128,1024))</c> (no activation between the three) that maps the GPT's second-pass latent into the
    /// semantic-codec embedding space. Lives in <c>s2mel.pth</c> (<c>net.gpt_layer.*</c>), not <c>gpt.pth</c>, so the
    /// caller passes the S2Mel dictionary. Required for <see cref="ComputeSecondPassLatent"/>.</summary>
    public void LoadGptLayer(IReadOnlyDictionary<string, Tensor> s2melWeights, string prefix)
    {
        _gptLayerW = new Tensor?[3];
        _gptLayerB = new Tensor?[3];
        for (int i = 0; i < 3; i++)
        {
            _gptLayerW[i] = WhisperOps.EnsureF32(s2melWeights[$"{prefix}.{i}.weight"]);
            _gptLayerB[i] = WhisperOps.EnsureF32(s2melWeights[$"{prefix}.{i}.bias"]);
        }
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
    /// (<c>inference_speech</c>): campplus mode is <c>cat([speakerConditioning + emoVec, zeros(1,2,hidden)], dim=1)</c>
    /// (one speaker+emotion slot plus two always-zero pad slots); conformer_perceiver mode is
    /// <c>cat([latent32 + emoVec, speed_emb(1), speed_emb(0)], dim=1)</c> (<c>speed_emb</c> is a real table lookup
    /// there, though the released checkpoint's rows are zero-initialised and the public API never varies them). Text
    /// gets <c>text_embedding[id] + text_pos_embedding[i]</c>, plus <c>lang_embedding[langId]</c> broadcast across
    /// every text position when <paramref name="langId"/> is given — campplus mode only (real
    /// <c>prepare_gpt_inputs</c>); conformer_perceiver mode has no language embedding, so <paramref name="langId"/>
    /// is ignored. No position embedding on the conditioning slots, matching the real source.</summary>
    public int[] Generate(IBackend backend, Tensor speakerConditioning, Tensor emoVec, int[] textTokenIds,
        int? langId, IndexTtsOptions options, ref uint rngState)
    {
        ThrowIfDisposed();
        int h = _cfg.Hidden;

        (Tensor conds, int condLen) = BuildConditioning(speakerConditioning, emoVec, h);
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
        using (Tensor textEmb = EmbedTextWithPositionAndLang(wrappedTextIds, Mode == IndexTts2SpeakerConditioning.Campplus ? langId : null, h))
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

    /// <summary>Assembles the GPT's conditioning prefix for the loaded <see cref="Mode"/>.</summary>
    private (Tensor Conds, int Length) BuildConditioning(Tensor speakerConditioning, Tensor emoVec, int h) =>
        Mode == IndexTts2SpeakerConditioning.Campplus
            ? (BuildCampplusConditioning(speakerConditioning, emoVec, h), 3)
            : (BuildConformerPerceiverConditioning(speakerConditioning, emoVec, _speedEmbed!, h), (int)speakerConditioning.Shape[1] + 2);

    /// <summary><c>cat([latents + emoVec.unsqueeze(1), speed_emb(1), speed_emb(0)], dim=1)</c> → <c>[1, L+2, hidden]</c>
    /// (<c>speed_emb(zeros)</c> is row 0, <c>speed_emb(ones)</c> is row 1, and the real concat order puts the
    /// "half" (row 1) slot first).</summary>
    internal static Tensor BuildConformerPerceiverConditioning(Tensor speakerLatents, Tensor emoVec, Tensor speedEmbed, int h)
    {
        int latentCount = (int)speakerLatents.Shape[1];
        Tensor conds = new(new TensorShape(1, latentCount + 2, h), DType.F32);
        float* cp = (float*)conds.DataPointer;
        float* sp = (float*)speakerLatents.DataPointer;
        float* ep = (float*)emoVec.DataPointer;
        float* speed = (float*)speedEmbed.DataPointer;
        for (int i = 0; i < latentCount; i++)
            for (int c = 0; c < h; c++)
                cp[(long)i * h + c] = sp[(long)i * h + c] + ep[c];
        for (int c = 0; c < h; c++)
        {
            cp[(long)latentCount * h + c] = speed[h + c];
            cp[(long)(latentCount + 1) * h + c] = speed[c];
        }
        return conds;
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

    /// <summary>IndexTTS-2.0's second GPT pass — the real <c>self.gpt(speech_conditioning_latent, text_tokens, ...,
    /// codes, ..., emo_vec=emovec, use_speed=zeros)</c> followed by <c>s2mel.gpt_layer</c>. Re-embeds the SAME
    /// conditioning prefix as <see cref="Generate"/> (the speaker latents and <paramref name="emoVec"/> are reused,
    /// not recomputed), the text as <c>[start, tokens…, stop]</c> from position 0 with no language embedding, and
    /// the mel as <c>[start, c1…c(T-1)]</c> (the reference feeds <c>[start, c1…cT, stop]</c> and drops the last two
    /// outputs — equivalent under the causal mask, one forward cheaper). Hidden states go through the backbone's
    /// <c>ln_f</c>, then <c>final_norm</c> (the reference's <c>get_logits</c> applies it), then the three-Linear
    /// <c>gpt_layer</c>. Returns <c>[1, T, 1024]</c> for <c>T = codes.Length</c>, to be summed with
    /// <see cref="HartsyInference.Audio.Models.Codecs.VocosFactorizedCodec.VqToEmbedding"/> of the same codes.</summary>
    public Tensor ComputeSecondPassLatent(IBackend backend, Tensor speakerConditioning, Tensor emoVec, int[] textTokenIds, int[] codes)
    {
        ThrowIfDisposed();
        RequireMode(IndexTts2SpeakerConditioning.ConformerPerceiver);
        if (_gptLayerW is null) throw new InvalidOperationException("gpt_layer weights not loaded — call LoadGptLayer first.");
        if (codes.Length == 0) throw new ArgumentException("codes must be non-empty.", nameof(codes));
        int h = _cfg.Hidden, t = codes.Length;

        int[] wrappedTextIds = new int[textTokenIds.Length + 2];
        wrappedTextIds[0] = StartTextToken;
        Array.Copy(textTokenIds, 0, wrappedTextIds, 1, textTokenIds.Length);
        wrappedTextIds[^1] = StopTextToken;
        int textLen = wrappedTextIds.Length;

        int[] melInput = new int[t];
        melInput[0] = StartMelToken;
        Array.Copy(codes, 0, melInput, 1, t - 1);

        (Tensor conds, int condLen) = BuildConditioning(speakerConditioning, emoVec, h);
        using (conds)
        using (Tensor textEmb = EmbedTextWithPositionAndLang(wrappedTextIds, langId: null, h))
        using (Tensor melEmb = EmbedWithPosition(melInput, _melEmbed!, _melPos!, h))
        using (Tensor condText = Concat(conds, textEmb, condLen, textLen, h))
        using (Tensor fullSeq = Concat(condText, melEmb, condLen + textLen, t, h))
        using (Tensor fullOut = _gpt.Forward(backend, fullSeq, nonCausal: false, cache: null, positionsApplied: true))
        using (Tensor melSlice = SliceSequence(fullOut, condLen + textLen, t, h))
        using (Tensor normed = new(melSlice.Shape, DType.F32))
        {
            backend.LayerNorm(normed, melSlice, _finalNormW!, _finalNormB!, 1e-5f);
            Tensor l1 = WhisperOps.ProjectLinear(backend, normed, _gptLayerW[0]!, _gptLayerB![0], 1, t, h, 256);
            Tensor l2 = WhisperOps.ProjectLinear(backend, l1, _gptLayerW[1]!, _gptLayerB[1], 1, t, 256, 128);
            l1.Dispose();
            Tensor l3 = WhisperOps.ProjectLinear(backend, l2, _gptLayerW[2]!, _gptLayerB[2], 1, t, 128, 1024);
            l2.Dispose();
            return l3;
        }
    }

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

    private static Tensor SliceSequence(Tensor x, int start, int length, int hidden)
    {
        Tensor outT = new(new TensorShape(1, length, hidden), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)start * hidden, (float*)outT.DataPointer,
            (long)length * hidden * 4, (long)length * hidden * 4);
        return outT;
    }

    private void RequireMode(IndexTts2SpeakerConditioning expected)
    {
        if (Mode != expected)
            throw new InvalidOperationException($"This operation needs a {expected} checkpoint, but the loaded one is {Mode}.");
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
            _spkEmbProjW, _spkEmbProjB, _langEmbed, _speedEmbed, _emovecLayerW, _emovecLayerB, _emoLayerW, _emoLayerB,
        ];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        if (_gptLayerW is not null && _gptLayerB is not null)
            for (int i = 0; i < 3; i++) { yield return _gptLayerW[i]!; yield return _gptLayerB[i]!; }
        if (_speakerEncoder is not null) foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
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
        _speakerEncoder?.Dispose();
        if (_ownedGptWeights is not null) foreach (Tensor t in _ownedGptWeights) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
