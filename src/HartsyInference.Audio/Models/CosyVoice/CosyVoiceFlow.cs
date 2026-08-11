using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2 flow-matching stage (speech tokens → mel). Mirrors
/// <c>cosyvoice/flow/flow.py:CausalMaskedDiffWithXvec</c>: embeds the LM speech tokens, runs them
/// through the (chunk-aware causal) encoder, time-upsamples 25 Hz → 50 Hz, projects to the 80-bin mel
/// conditioning <c>μ</c>, projects the CAM++ speaker vector to mel dim, then solves the OT-CFM ODE with
/// classifier-free guidance to produce the target mel.
///
/// <para>The token-conditioning path is now the real <see cref="UpsampleConformerEncoder"/>: embedded
/// tokens → conformer stack → 2× ConvTranspose1d upsample (25 Hz → 50 Hz) → conformer stack →
/// <c>encoder_proj</c> → mel conditioning <c>μ</c>. Everything downstream of <c>μ</c> (the CFM solve, CFG,
/// speaker conditioning) is exact.</para></summary>
public sealed unsafe class CosyVoiceFlow : IDisposable
{
    private readonly CosyVoiceConfig _cfg;
    private readonly CausalConditionalDecoder _estimator;
    private readonly ConditionalCfm _cfm;
    private readonly UpsampleConformerEncoder _encoder;
    private int _disposed;

    private Tensor? _inputEmbedding;     // [speechVocab, inputSize]
    private Tensor? _encoderProjW, _encoderProjB;   // encoderOutputSize → melBins
    private Tensor? _spkAffineW, _spkAffineB;        // 192 → melBins

    public CosyVoiceFlow(CosyVoiceConfig cfg)
    {
        _cfg = cfg;
        _estimator = new CausalConditionalDecoder(cfg.Flow);
        _cfm = new ConditionalCfm(_estimator, cfg.Flow.MelBins);
        _encoder = new UpsampleConformerEncoder(cfg.Flow.EncoderOutputSize, cfg.Flow.EncoderNumHeads,
            cfg.Flow.EncoderNumPreBlocks, cfg.Flow.EncoderNumPostBlocks);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inputEmbedding = WhisperOps.EnsureF32(w[$"{p}input_embedding.weight"]);
        _encoder.LoadWeights(w, $"{p}encoder");
        _encoderProjW = WhisperOps.EnsureF32(w[$"{p}encoder_proj.weight"]);
        _encoderProjB = WhisperOps.EnsureF32(w[$"{p}encoder_proj.bias"]);
        _spkAffineW = WhisperOps.EnsureF32(w[$"{p}spk_embed_affine_layer.weight"]);
        _spkAffineB = WhisperOps.EnsureF32(w[$"{p}spk_embed_affine_layer.bias"]);
        _estimator.LoadWeights(w, $"{p}decoder.estimator");
    }

    /// <summary>Generates the target mel <c>[1, melBins, T_mel]</c> for a speech-token stream.
    /// <paramref name="promptSpeechTokens"/> + <paramref name="promptMel"/> are the reference clip's
    /// tokens + mel (empty for preset-voice modes); <paramref name="speakerEmbed"/> is the CAM++
    /// 192-d vector.</summary>
    /// <param name="chunkCausalSize">When set, solves with a block-diagonal-causal attention mask
    /// (<see cref="MaskBuilder.BuildChunkCausalMask"/>) over the whole <c>tMel</c> span instead of the
    /// default full-attention mask — the Phase-5.0 quality probe: exercises the chunk-causal training mode
    /// in one monolithic call, no actual chunking/state-carry yet. Null preserves today's exact output.</param>
    /// <param name="x0Override">When set, replaces the fresh <paramref name="seed"/>-derived CFM noise draw
    /// (see <see cref="ConditionalCfm.Solve"/>'s doc comment) — used by <see cref="InferenceChunk"/> so each
    /// chunk's target frames draw the SAME noise a monolithic call would give those absolute positions,
    /// instead of a fresh unrelated draw every call.</param>
    public Tensor Inference(IBackend backend,
        ReadOnlySpan<int> speechTokens,
        ReadOnlySpan<int> promptSpeechTokens,
        Tensor? promptMel,
        Tensor speakerEmbed,
        int seed = 0,
        int? chunkCausalSize = null,
        Tensor? x0Override = null)
    {
        if (_inputEmbedding is null) throw new InvalidOperationException("CosyVoiceFlow weights not loaded.");
        int inputSize = _cfg.Flow.InputSize;
        int mel = _cfg.Flow.MelBins;

        // Concatenate prompt + target speech tokens and embed.
        int nTok = promptSpeechTokens.Length + speechTokens.Length;
        Tensor tokEmb = new(new TensorShape(1, nTok, inputSize), DType.F32);
        int row = 0;
        for (int i = 0; i < promptSpeechTokens.Length; i++) WriteEmbRow(tokEmb, row++, promptSpeechTokens[i], inputSize);
        for (int i = 0; i < speechTokens.Length; i++) WriteEmbRow(tokEmb, row++, speechTokens[i], inputSize);

        // UpsampleConformerEncoder: conformer stack → 2× time upsample (25 Hz token → 50 Hz) → conformer stack.
        Tensor up = _encoder.Forward(backend, tokEmb, inputSize);
        tokEmb.Dispose();
        int tMel = (int)up.Shape[1];
        int encOut = (int)up.Shape[2];

        // encoder_proj → μ [1, T_mel, mel] then transpose to channels-first [1, mel, T_mel].
        Tensor muSeq = WhisperOps.ProjectLinear(backend, up, _encoderProjW!, _encoderProjB, 1, tMel, encOut, mel);
        up.Dispose();
        Tensor mu = new(new TensorShape(1, mel, tMel), DType.F32);
        backend.Transpose2D(mu, muSeq, tMel, mel);
        muSeq.Dispose();

        // Speaker embedding → mel dim [1, mel] (kept as [1, mel, 1] for broadcast). The CAM++ x-vector MUST be
        // L2-normalized before the affine projection — reference flow.inference does `F.normalize(embedding, dim=1)`
        // (the affine was trained on unit-norm inputs; feeding the raw ~10-30×-magnitude x-vector over-scales the
        // speaker conditioning and yields a growly/demonic voice).
        Tensor spkNorm = L2NormalizeRow(speakerEmbed, _cfg.Flow.SpeakerEmbedDim);
        Tensor spk = WhisperOps.ProjectLinear(backend, spkNorm, _spkAffineW!, _spkAffineB, 1, 1, _cfg.Flow.SpeakerEmbedDim, mel);
        spkNorm.Dispose();
        Tensor spkChan = spk.Reshape(new TensorShape(1, mel, 1));

        // Reference-mel conditioning: place the prompt mel in the prefix, zeros elsewhere.
        Tensor cond = new(new TensorShape(1, mel, tMel), DType.F32);
        if (promptMel is not null) WritePromptCond(cond, promptMel, mel, tMel);

        Tensor? attnMask = chunkCausalSize is > 0 ? MaskBuilder.BuildChunkCausalMask(tMel, chunkCausalSize.Value) : null;
        Tensor outMel = _cfm.Solve(backend, mu, spkChan, cond, _cfg.Flow.NumEulerSteps, _cfg.Flow.CfgRate, seed, attnMask, x0Override);
        attnMask?.Dispose();
        mu.Dispose();
        spk.Dispose();
        cond.Dispose();

        // The solve produces mel for the whole [prompt ++ target] token span (the prompt region is pinned to the
        // reference mel via `cond`). The reference returns only the generated tail — trim the prompt frames so the
        // vocoder doesn't replay the reference clip before the target speech.
        int promptFrames = Math.Min(promptSpeechTokens.Length * UpsampleConformerEncoder.TokenMelRatio, tMel);
        if (promptFrames <= 0) return outMel;
        Tensor tail = TrimPromptFrames(outMel, mel, tMel, promptFrames);
        outMel.Dispose();
        return tail;
    }

    /// <summary>Returns a fresh <c>[1, mel, tMel-promptFrames]</c> holding the generated tail of a channels-first
    /// mel, dropping the leading prompt region.</summary>
    private static Tensor TrimPromptFrames(Tensor full, int mel, int tMel, int promptFrames)
    {
        int keep = tMel - promptFrames;
        Tensor tail = new(new TensorShape(1, mel, keep), DType.F32);
        float* sp = (float*)full.DataPointer;
        float* dp = (float*)tail.DataPointer;
        for (int c = 0; c < mel; c++)
            Buffer.MemoryCopy(sp + (long)c * tMel + promptFrames, dp + (long)c * keep, (long)keep * 4, (long)keep * 4);
        return tail;
    }

    /// <summary>Chunked variant of <see cref="Inference"/> for streaming: produces mel for ONE chunk of new
    /// speech tokens, using the caller-supplied lookback (previous chunk's own settled tokens + mel, or the
    /// real zero-shot reference clip for chunk 0) as conditioning context instead of recomputing from t=0
    /// every call. This is a thin wrapper over <see cref="Inference"/> — its existing prompt/trim mechanism
    /// (<see cref="WritePromptCond"/>/<see cref="TrimPromptFrames"/>) already IS the lookback mechanism this
    /// needs; the "prompt" is just the previous chunk's tail instead of a reference clip.
    ///
    /// <para><b>NOT PRODUCTION-VIABLE AS-IS (found + measured 2026-08-11, real weights, real generation)</b>:
    /// rolling the model's OWN previous output forward as the next call's <c>cond</c> prompt (exactly as
    /// implemented below, and exactly as the plan's §5.2 text specifies) causes a real, PROGRESSIVE mel-domain
    /// mean-level drift. Chunk 0's returned-mel mean/std (-3.68/1.48) is close to the matching monolithic-mel
    /// range (-3.90/1.55 — chunk 0's lookback is the real reference clip, no self-conditioning has happened
    /// yet); by chunk 8 the mean has drifted to -3.23 vs the monolithic range's -4.23 (std stays close: 2.06
    /// vs 1.98 — this is a LEVEL shift, not a structural/spectral-shape corruption). Because mel is log-scale,
    /// that ~1.0-nat mean drift compounds roughly exponentially in the audio domain: measured per-second RMS
    /// ratio (chunked ÷ monolithic) grew 1.34× → 2.72× → 2.90× → 4.63× → 5.61× → 7.16× → 3.98× → 5.78× → 7.46×
    /// over a 9.36s utterance (15-token chunks, 3-token lookahead), clipping near ±1.0 well before the end.
    /// Growing the lookback to be cumulative (MORE self-generated content per call, not less) made it WORSE
    /// (RMS ratio ~4.5× vs ~2.2× for this bounded 1-chunk-lookback design) — this rules out context-length
    /// truncation as the cause (growing the window would have fixed it) and confirms the mechanism is
    /// exposure bias from feeding the model's own generated mel back in as <c>cond</c>, which CosyVoice2's
    /// flow decoder was (almost certainly) only ever trained to receive as REAL reference-clip mel. Content
    /// stays correct throughout (Whisper transcribes chunked output word-perfect against the target text at
    /// every drift level measured) — this is a level/energy bug, not a corruption bug, but it is real and
    /// would make longer streamed utterances progressively louder/more distorted. A pre-drawn full-utterance
    /// noise buffer sliced per chunk (<c>x0Override</c>/<paramref name="fullNoise"/>, tried first as the
    /// obvious suspect) does NOT fix this — the drift is in <c>cond</c>'s conditioning distribution, not the
    /// CFM noise seed.
    ///
    /// <para><b>Design lever for whoever picks this back up</b> (not implemented — a genuinely different
    /// design from what's below): keep <c>cond</c>'s prompt region pinned to the REAL reference-clip mel for
    /// every chunk (never self-generated), and use only the TOKEN-domain lookback (which the encoder half of
    /// this method already does) for local continuity. That breaks the self-conditioning feedback loop
    /// entirely, at the cost of the CFM losing direct mel-domain continuity with the immediately preceding
    /// generated chunk — whether that trades this drift bug for a NEW boundary-discontinuity artifact is
    /// untested, and is the real next empirical question, not an assumption to build on top of.</para>
    ///
    /// <para><b>Masking is OFF here</b> (Phase 5.2), and stays off permanently for the encoder half even
    /// after Phase 5.3 turns masking on for the CFM decoder: <see cref="UpsampleConformerEncoder"/>'s
    /// <c>RelPosAttention</c> has no mask parameter at all — it is unconditionally full/bidirectional over
    /// whatever span it's fed. This method's ONLY streaming mechanism for the encoder is windowed recompute
    /// (feed a bounded lookback+chunk+lookahead span, discard the rest), never a mask.</para>
    ///
    /// <para><b>Lookahead, not just lookback</b>: <c>pre_lookahead_layer.conv1</c>'s right-pad means the
    /// encoder needs <c>k1-1</c> real tokens beyond a chunk's own boundary to produce a settled result for
    /// it (read <c>k1</c> from the checkpoint via the encoder's own weights, e.g. 4 for the current
    /// CosyVoice2-0.5B <c>flow.pt</c>, i.e. a 3-token floor — never hardcode it). <paramref name="chunkTokens"/>
    /// ++ <paramref name="lookaheadTokens"/> are BOTH fed to the encoder (needed for <paramref
    /// name="chunkTokens"/>'s own result to be settled), but only <paramref name="chunkTokens"/>'s span of
    /// mel is returned — the lookahead tokens' own mel is discarded here and recomputed properly once the
    /// caller's cursor reaches them as the START of a LATER chunk. Nothing is wasted: those same token ids
    /// become that later chunk's own leading content, not lookback context.</para>
    ///
    /// <para>Chunk-to-chunk rolling is the caller's job: after each call, the caller should carry
    /// <paramref name="chunkTokens"/> (NOT including <paramref name="lookaheadTokens"/>) and this call's
    /// returned mel forward as the NEXT call's <paramref name="lookbackTokens"/>/<c>lookbackMel</c> — by
    /// construction the returned mel's frame count is always exactly <c>chunkTokens.Length ×
    /// <see cref="UpsampleConformerEncoder.TokenMelRatio"/></c>, so lookback token-count and lookback
    /// mel-frame-count stay exactly in sync across chunks (no frame-count truncation heuristics needed —
    /// see the mean-LEVEL drift warning above, a separate issue from frame-count bookkeeping).</para>
    ///
    /// <para><b>CFM noise must be pre-drawn for the WHOLE utterance and sliced, not re-seeded per call</b> —
    /// see <see cref="ConditionalCfm.Solve"/>'s <c>x0Override</c> doc. Because <paramref name="lookbackTokens"/>
    /// is always exactly the immediately-preceding chunk's own tokens (chunk 0's lookback is the reference
    /// clip's own prompt tokens, which is exactly where the monolithic call's own token layout starts too),
    /// the span [lookback ++ chunk ++ lookahead] fed to any one call is always a CONTIGUOUS absolute range of
    /// the full utterance's token timeline — <paramref name="absoluteTokenOffset"/> is that range's start
    /// (in tokens, from the same origin as <c>promptSpeechTokens ++ speechTokens</c>), and
    /// <paramref name="fullNoise"/>/<paramref name="fullNoiseFrames"/> are the one-time
    /// <see cref="ConditionalCfm.DrawFullNoise"/> draw covering the whole utterance.</para></summary>
    public Tensor InferenceChunk(IBackend backend,
        ReadOnlySpan<int> lookbackTokens, Tensor lookbackMel,
        ReadOnlySpan<int> chunkTokens, ReadOnlySpan<int> lookaheadTokens,
        Tensor speakerEmbed, int seed,
        Tensor fullNoise, int fullNoiseFrames, int absoluteTokenOffset)
    {
        int lookahead = lookaheadTokens.Length;
        int[] combined = new int[chunkTokens.Length + lookahead];
        chunkTokens.CopyTo(combined);
        lookaheadTokens.CopyTo(combined.AsSpan(chunkTokens.Length));

        int mel = _cfg.Flow.MelBins;
        int spanTokens = lookbackTokens.Length + combined.Length;
        int spanFrames = spanTokens * UpsampleConformerEncoder.TokenMelRatio;
        int frameOffset = absoluteTokenOffset * UpsampleConformerEncoder.TokenMelRatio;
        Tensor x0 = ConditionalCfm.SliceNoise(fullNoise, mel, fullNoiseFrames, frameOffset, spanFrames);

        Tensor full = Inference(backend, combined, lookbackTokens, lookbackMel, speakerEmbed, seed, chunkCausalSize: null, x0Override: x0);
        x0.Dispose();
        if (lookahead <= 0) return full;

        // `full` is already trimmed of the lookback/prompt region by Inference's own TrimPromptFrames — it
        // holds mel for [chunkTokens ++ lookaheadTokens] exactly (Upsample guarantees tUp = t·ratio exactly,
        // no rounding). Crop off the lookahead's own trailing frames before returning.
        int totalFrames = (int)full.Shape[2];
        int lookaheadFrames = lookahead * UpsampleConformerEncoder.TokenMelRatio;
        int keep = totalFrames - lookaheadFrames;
        if (keep <= 0) return full;
        Tensor chunkMel = TrimTrailingFrames(full, mel, totalFrames, keep);
        full.Dispose();
        return chunkMel;
    }

    /// <summary>Returns a fresh <c>[1, mel, keep]</c> holding the LEADING <paramref name="keep"/> frames of a
    /// channels-first mel, dropping the trailing frames — the mirror of <see cref="TrimPromptFrames"/>.</summary>
    private static Tensor TrimTrailingFrames(Tensor full, int mel, int totalFrames, int keep)
    {
        Tensor head = new(new TensorShape(1, mel, keep), DType.F32);
        float* sp = (float*)full.DataPointer;
        float* dp = (float*)head.DataPointer;
        for (int c = 0; c < mel; c++)
            Buffer.MemoryCopy(sp + (long)c * totalFrames, dp + (long)c * keep, (long)keep * 4, (long)keep * 4);
        return head;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_inputEmbedding, _encoderProjW, _encoderProjB, _spkAffineW, _spkAffineB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _estimator.EnumerateWeights()) yield return t;
    }

    private void WriteEmbRow(Tensor dst, int row, int token, int dim)
    {
        int vocab = (int)_inputEmbedding!.Shape[0];
        if ((uint)token >= (uint)vocab) throw new ArgumentException($"speech token {token} out of range [0, {vocab}).");
        float* sp = (float*)_inputEmbedding.DataPointer + (long)token * dim;
        float* dp = (float*)dst.DataPointer + (long)row * dim;
        Buffer.MemoryCopy(sp, dp, dim * 4, dim * 4);
    }

    /// <summary>L2-normalizes the speaker x-vector into a fresh <c>[1, 1, dim]</c> tensor
    /// (<c>F.normalize(embedding, dim=1)</c>, eps 1e-12). Never mutates the caller's tensor.</summary>
    private static Tensor L2NormalizeRow(Tensor v, int dim)
    {
        if (v.ElementCount != dim) throw new ArgumentException($"speaker embed must have {dim} elements, got {v.ElementCount}.");
        Tensor outT = new(new TensorShape(1, 1, dim), DType.F32);
        float* sp = (float*)v.DataPointer;
        float* dp = (float*)outT.DataPointer;
        double sum = 0;
        for (int i = 0; i < dim; i++) sum += (double)sp[i] * sp[i];
        float inv = 1f / MathF.Max((float)Math.Sqrt(sum), 1e-12f);
        for (int i = 0; i < dim; i++) dp[i] = sp[i] * inv;
        return outT;
    }

    private static void WritePromptCond(Tensor cond, Tensor promptMel, int mel, int tMel)
    {
        int tp = Math.Min((int)promptMel.Shape[2], tMel);
        float* pp = (float*)promptMel.DataPointer;
        float* cp = (float*)cond.DataPointer;
        for (int c = 0; c < mel; c++)
            for (int j = 0; j < tp; j++)
                cp[(long)c * tMel + j] = pp[(long)c * (int)promptMel.Shape[2] + j];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
