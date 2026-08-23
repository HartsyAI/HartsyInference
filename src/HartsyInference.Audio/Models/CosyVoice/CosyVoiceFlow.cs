using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2 flow-matching stage (speech tokens → mel).</summary>
/// <remarks>Mirrors
/// <c>cosyvoice/flow/flow.py:CausalMaskedDiffWithXvec</c>: embeds the LM speech tokens, runs them
/// through the (chunk-aware causal) encoder, time-upsamples 25 Hz → 50 Hz, projects to the 80-bin mel
/// conditioning <c>μ</c>, projects the CAM++ speaker vector to mel dim, then solves the OT-CFM ODE with
/// classifier-free guidance to produce the target mel.
///
/// <para>The token-conditioning path is now the real <see cref="UpsampleConformerEncoder"/>: embedded
/// tokens → conformer stack → 2× ConvTranspose1d upsample (25 Hz → 50 Hz) → conformer stack →
/// <c>encoder_proj</c> → mel conditioning <c>μ</c>. Everything downstream of <c>μ</c> (the CFM solve, CFG,
/// speaker conditioning) is exact.</para></remarks>
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

    /// <summary>Generates the target mel <c>[1, melBins, T_mel]</c> for a speech-token stream. <paramref name="promptSpeechTokens"/> + <paramref name="promptMel"/> are the reference clip's tokens + mel (empty for preset-voice modes); <paramref name="speakerEmbed"/> is the CAM++ 192-d vector.</summary>
    /// <param name="chunkCausalSize">When set, solves with a block-diagonal-causal attention mask (<see cref="MaskBuilder.BuildChunkCausalMask"/>) over the whole <c>tMel</c> span instead of the default full-attention mask — the Phase-5.0 quality probe: exercises the chunk-causal training mode in one monolithic call, no actual chunking/state-carry yet. Null preserves today's exact output.</param>
    /// <param name="x0Override">When set, replaces the fresh <paramref name="seed"/>-derived CFM noise draw (see <see cref="ConditionalCfm.Solve"/>'s doc comment) — used by the streaming path so each call's target frames draw the SAME noise a monolithic call would give those absolute positions, instead of a fresh unrelated draw every call.</param>
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

    /// <summary>Returns a fresh <c>[1, mel, tMel-promptFrames]</c> holding the generated tail of a channels-first mel, dropping the leading prompt region.</summary>
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

    /// <summary>Returns a fresh <c>[1, mel, len]</c> holding <c>full[:, :, start..start+len)</c> of a channels-first mel.</summary>
    private static Tensor SliceFrames(Tensor full, int mel, int totalFrames, int start, int len)
    {
        Tensor slice = new(new TensorShape(1, mel, len), DType.F32);
        float* sp = (float*)full.DataPointer;
        float* dp = (float*)slice.DataPointer;
        for (int c = 0; c < mel; c++)
            Buffer.MemoryCopy(sp + (long)c * totalFrames + start, dp + (long)c * len, (long)len * 4, (long)len * 4);
        return slice;
    }

    /// <summary>Streaming inference over a BOUNDED window of the target-token history (see remarks for the measured cost/quality tradeoffs and the one known artifact).</summary>
    /// <remarks>The key structural fact making this safe: in the
    /// EXISTING, already-verified monolithic <see cref="Inference"/> path, <c>cond</c> is zero for the ENTIRE
    /// target-token span — <see cref="WritePromptCond"/> only ever writes the real
    /// <paramref name="promptMel"/> into the reference-clip prefix, nothing else. So a windowed PAST-target-
    /// token span occupies the exact same zero-cond region here that it would in a monolithic call; there is
    /// no new self-conditioning channel for drift to compound through, regardless of window size — only the
    /// ENCODER's attention span (and therefore compute cost) shrinks.
    ///
    /// <para><paramref name="windowTokens"/> is a BOUNDED slice of the target-token history — NOT the full
    /// history from token 0 — ending at the current chunk boundary.
    /// <paramref name="windowStartToken"/> is that slice's absolute start (0-based into the full target-token
    /// sequence, i.e. the same origin as <c>promptSpeechTokens ++ speechTokens</c>). Because the
    /// window's own absolute position is NOT adjacent to the real prompt's absolute position once
    /// <paramref name="windowStartToken"/> &gt; 0 (there's a gap: the earlier target tokens excluded from this
    /// call's window), the CFM noise for this call is built from TWO separate slices of
    /// <paramref name="fullNoise"/> — <c>[0, promptFrames)</c> (always the same, matches the real prompt's
    /// own fixed absolute position) concatenated with <c>[windowStartToken×ratio&#43;promptFrames,
    /// …&#43;windowFrames)</c> (the window's own absolute position) — NOT a single contiguous slice. This keeps
    /// noise-per-absolute-frame consistent across calls with DIFFERENT window starts, isolating the encoder's
    /// own windowing effect from noise-seed drift.</para>

    /// <para><b>What <paramref name="marginFrames"/> is for, measured not guessed (2026-08-11)</b>: because
    /// <see cref="UpsampleConformerEncoder"/>'s attention is fully bidirectional (it has no mask parameter at
    /// all — windowed recompute is the ONLY streaming mechanism available for the encoder half), a frame's own
    /// encoder output DOES shift slightly when more future tokens arrive in a later call. Measured via a real
    /// two-call experiment (120 vs 234 real LM tokens, same prompt): per-frame max-abs-diff over the
    /// overlapping range is a modest, NON-decaying floor of ~0.1–0.6 through most of the range (5-30% relative
    /// to typical mel std ~1.5-2.7), escalating sharply to 1.6–4.5 within the last ~20-30 frames nearest the
    /// shorter call's own live edge. Frames beyond <paramref name="marginFrames"/> from the live edge are
    /// treated as settled enough to emit; frames within it are recomputed (not yet emitted) on the NEXT call.
    /// This floor does NOT compound across chunks — every call is independently grounded in the real prompt,
    /// so an early chunk's small settling error never feeds into a later chunk's.</para>
    ///
    /// <para><b>Measured result (2026-08-11), best config windowSizeTokens=150/chunkSizeTokens=25/
    /// marginFrames=40, 26.2s test utterance — VIABLE, ONE KNOWN QUANTIFIED ARTIFACT, ACCEPTED (not a bug
    /// to keep chasing — see the parameter sweep below)</b>: per-call wall-clock FLATTENS once the window
    /// fills, 3.45× real-time overall. Mel relL2 vs monolithic ≈ 5.6-5.9% across every config tested, and
    /// per-second audio RMS ratio stayed bounded with no growth trend over 26s. The two rejected alternatives
    /// this replaced (unbounded recompute at 13.75× real time, and self-conditioned chunking whose mel level
    /// drift compounded to a 7.46× RMS ratio) are recorded with their measurements in
    /// <c>docs/Checklists/MODEL_STATUS_AUDIO.md</c> under CosyVoice 2.</para>
    ///
    /// <para><b>The one artifact, fully characterized via a 5-way parameter sweep, not guessed at</b>: Whisper
    /// cross-transcript comparison found "quick brown fox" → "quit brown fox" (one word out of ~80, trailing
    /// "-ck" clipped). Swept windowSizeTokens ∈ {150, 200, 300} × marginFrames ∈ {40, 60} × chunkSizeTokens ∈
    /// {15, 25} — EVERY bounded combination reproduces the identical substitution, byte-for-byte, in both the
    /// flow-only and full-pipeline variants. A decisive discriminator settled the mechanism: an UNBOUNDED
    /// recompute over the full ~600-token history (not a sliding window) transcribes this word
    /// CORRECTLY on the exact same utterance/seed. So the substitution is not a margin or chunk-size tuning
    /// problem — it requires near-COMPLETE history to resolve, which no practical bounded window provides
    /// (tested up to 300 tokens, half the utterance, with zero improvement over 150). Most likely mechanism:
    /// the LM's own sampled speech tokens for this specific word are a borderline/ambiguous realization that
    /// only fully resolves with (near-)complete bidirectional context — an isolated, adversarial case, not a
    /// general quality collapse (everything else in the ~80-word utterance transcribes perfectly across every
    /// config, including a shared unrelated Whisper mishearing present identically in ALL variants tested,
    /// bounded and unbounded alike). Accepted as a known, quantified (~1.25% WER, isolated, non-cascading)
    /// limitation of the bounded-window design rather than continuing to chase a fix that would require
    /// reintroducing the unbounded design's disqualifying cost.</para>
    ///
    /// <para><paramref name="chunkCausalSize"/> (Phase 5.3): optional chunk-causal mask for the CFM/mel-
    /// decoder half only (<see cref="CausalConditionalDecoder"/>, via <see cref="Inference"/>'s own param) —
    /// the token encoder can never accept one (see the unmaskable-encoder note above). Null (default)
    /// preserves this method's original unmasked behavior exactly.</para>
    ///
    /// <para><b>Measured (2026-08-11)</b>, chunkCausalSize=50 (matches the tuned chunkSizeTokens×ratio) on
    /// the same 26.2s utterance/config as the sweep above: ~14% FASTER (real-time factor improved) at a
    /// small quality cost (mel relL2 6.5% vs 5.9% unmasked). Whisper transcript is BYTE-IDENTICAL to the
    /// unmasked case, including the known "quick"→"quit" artifact — masking the CFM decoder's attention
    /// does NOT touch that artifact at all, confirming it originates in the (unmaskable) encoder, not CFM
    /// attention. Real, modest speed/quality tradeoff knob for callers who want it — left off by default
    /// (best measured quality) rather than defaulted on, since the speed win is smaller than
    /// <paramref name="chunkSizeTokens"/> tuning's own effect and stacks with it if wanted.</para></remarks>
    public Tensor InferenceGrowingWindowed(IBackend backend,
        ReadOnlySpan<int> windowTokens, int windowStartToken,
        ReadOnlySpan<int> promptSpeechTokens, Tensor promptMel,
        Tensor speakerEmbed, int seed,
        Tensor fullNoise, int fullNoiseFrames, int marginFrames,
        ref int emittedFrames, bool isFinal,
        int? chunkCausalSize = null)
    {
        int mel = _cfg.Flow.MelBins;
        int ratio = UpsampleConformerEncoder.TokenMelRatio;
        int promptFrames = promptSpeechTokens.Length * ratio;
        int windowFrames = windowTokens.Length * ratio;
        int spanFrames = promptFrames + windowFrames;

        Tensor x0;
        if (windowStartToken == 0)
        {
            // Window starts at token 0 -> contiguous with the prompt region.
            x0 = ConditionalCfm.SliceNoise(fullNoise, mel, fullNoiseFrames, 0, spanFrames);
        }
        else
        {
            int windowFrameOffset = (promptSpeechTokens.Length + windowStartToken) * ratio;
            Tensor promptPart = ConditionalCfm.SliceNoise(fullNoise, mel, fullNoiseFrames, 0, promptFrames);
            Tensor windowPart = ConditionalCfm.SliceNoise(fullNoise, mel, fullNoiseFrames, windowFrameOffset, windowFrames);
            x0 = new Tensor(new TensorShape(1, mel, spanFrames), DType.F32);
            float* pp = (float*)promptPart.DataPointer;
            float* wp = (float*)windowPart.DataPointer;
            float* xp = (float*)x0.DataPointer;
            for (int c = 0; c < mel; c++)
            {
                Buffer.MemoryCopy(pp + (long)c * promptFrames, xp + (long)c * spanFrames, promptFrames * 4L, promptFrames * 4L);
                Buffer.MemoryCopy(wp + (long)c * windowFrames, xp + (long)c * spanFrames + promptFrames, windowFrames * 4L, windowFrames * 4L);
            }
            promptPart.Dispose();
            windowPart.Dispose();
        }

        Tensor full = Inference(backend, windowTokens, promptSpeechTokens, promptMel, speakerEmbed, seed,
            chunkCausalSize: chunkCausalSize, x0Override: x0);
        x0.Dispose();

        int totalFrames = (int)full.Shape[2];
        // emittedFrames is a GLOBAL cursor (absolute mel frames emitted so far), but `full` is LOCAL to this
        // window (starts at windowStartToken's own frame position) -- convert between the two.
        int windowStartFrame = windowStartToken * ratio;
        int localEmitted = Math.Max(0, emittedFrames - windowStartFrame);
        int settledEnd = isFinal ? totalFrames : Math.Max(localEmitted, totalFrames - marginFrames);
        if (settledEnd <= localEmitted)
        {
            full.Dispose();
            return new Tensor(new TensorShape(1, mel, 0), DType.F32);
        }
        Tensor newPortion = SliceFrames(full, mel, totalFrames, localEmitted, settledEnd - localEmitted);
        full.Dispose();
        emittedFrames = windowStartFrame + settledEnd;
        return newPortion;
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

    /// <summary>L2-normalizes the speaker x-vector into a fresh <c>[1, 1, dim]</c> tensor (<c>F.normalize(embedding, dim=1)</c>, eps 1e-12). Never mutates the caller's tensor.</summary>
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
