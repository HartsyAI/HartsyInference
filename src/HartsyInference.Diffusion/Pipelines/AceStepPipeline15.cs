using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>ACE-Step v1.5 turbo text/lyrics-to-music pipeline — the 2B Qwen3-block flow-matching DiT over Oobleck latents (25 Hz, 64-ch): pre-computed Qwen3-Embedding-0.6B prompt + lyric states → packed [lyric ‖ timbre ‖ text] cross-attention conditions → fixed 8-step shift-3 Euler ODE (hardcoded <c>SHIFT_TIMESTEPS</c> table, <b>no CFG</b>, r = t) → Oobleck decode to 48 kHz stereo. Plain T2M uses the silence latent as <c>src_latents</c> (VAE-encoded at first use when the injected decoder can encode — the reference ships it as <c>silence_latent.pt</c>; recompute is the research doc's sanctioned alternative) with an all-ones chunk mask. The FSQ/LM hint path (cover mode, Comfy's <c>generate_audio_codes</c>) is phase 2 — <c>lmHints</c> is its null-default extension point. <b>Status: DiT + condition encoder numerically verified vs the f32 torch oracle on the real turbo checkpoint</b> (condition encoder maxAbs 1.9e-6; DiT velocity / full-loop latent corr 1.0 — see <c>AceStep15DitParityTests</c>); Oobleck VAE decode verified corr 0.9999999999. Hints-less quality vs Comfy defaults remains an open question (research §7).</summary>
/// <summary>Per-generation knobs for <see cref="AceStepPipeline15.Generate"/>. Defaults mirror the upstream per-family UI config: turbo ignores guidance (distilled, CFG baked in) and enables DCW; base/sft run CFG (APG default, ADG opt-in, interval-gated) with DCW off.</summary>
public sealed record AceStep15GenerateOptions
{
    public float? Shift { get; init; }
    public int? Seed { get; init; }
    /// <summary>Steps; null = config default (turbo 8; callers pass 50 sft / 32 base).</summary>
    public int? InferSteps { get; init; }
    /// <summary>CFG strength; ≤1 (or a turbo config) disables the uncond pass.</summary>
    public float GuidanceScale { get; init; } = 1f;
    /// <summary>ADG instead of the default APG blend. Deprecated: honored only when <see cref="GuidanceType"/> is empty.</summary>
    public bool UseAdg { get; init; }

    /// <summary>Guidance blend: "apg" (default), "cfg" (plain classifier-free), or "adg".</summary>
    public string GuidanceType { get; init; } = "";
    /// <summary>Guidance applies only when <c>cfgIntervalStart ≤ t ≤ cfgIntervalEnd</c> (momentum skips too).</summary>
    public float CfgIntervalStart { get; init; } = 0f;
    public float CfgIntervalEnd { get; init; } = 1f;
    /// <summary>"ode" (Euler) or "sde" (predict clean + renoise at linear next-t).</summary>
    public string InferMethod { get; init; } = "ode";
    /// <summary>DCW wavelet correction; null = upstream default (on for turbo, off for base/sft).</summary>
    public bool? DcwEnabled { get; init; }
    public string DcwMode { get; init; } = "double";
    public float DcwScaler { get; init; } = 0.05f;
    public float DcwHighScaler { get; init; } = 0.02f;
}

public sealed unsafe class AceStepPipeline15 : DiffusionPipelineBase
{
    private readonly AceStep15Dit _dit;
    private readonly AceStep15ConditionEncoder _encoder;
    private readonly IAudioLatentDecoder _vae;
    private readonly AceStep15Config _config;
    private Tensor? _silenceFrames;
    private Tensor? _nullConditionEmb;   // learned [1,1,encH] uncond row (base/sft CFG)

    public AceStepPipeline15(IBackend backend, AceStep15Dit dit, AceStep15ConditionEncoder encoder,
        IAudioLatentDecoder vae, AceStep15Config config)
        : base(backend)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>Sets the checkpoint's learned <c>null_condition_emb</c> <c>[1, 1, encHidden]</c> — the CFG unconditional row substituted for every position of the packed condition sequence (base/sft).</summary>
    public void SetNullConditionEmb(Tensor nullEmb)
    {
        ThrowIfDisposed();
        _nullConditionEmb?.Dispose();
        _nullConditionEmb = nullEmb;
    }

    /// <summary>Sets the shipped <c>silence_latent.pt</c> rows <c>[T, 64]</c> as the text-to-music src latent (upstream loads it per checkpoint; byte-identical across all v1.5 repos). When unset, the pipeline falls back to VAE-encoding digital silence.</summary>
    public void SetSilenceLatent(Tensor silenceRows)
    {
        ThrowIfDisposed();
        if (silenceRows.Shape.Rank != 2 || (int)silenceRows.Shape[1] != _config.LatentChannels)
            throw new ArgumentException($"silence latent must be [T, {_config.LatentChannels}]; got {silenceRows.Shape}.", nameof(silenceRows));
        _silenceFrames?.Dispose();
        _silenceFrames = silenceRows;
    }

    /// <summary>Generates stereo audio from pre-computed Qwen3-Embedding-0.6B prompt states <c>[T_text, 1024]</c> and optional lyric states <c>[L, 1024]</c> (null → no lyric tokens; instrumental via prompt tags). <paramref name="timbreLatent"/> is an optional reference-audio Oobleck latent <c>[T_ref, 64]</c>; <paramref name="lmHints"/> is the phase-2 cover-mode hook (25 Hz detokenizer latents <c>[1, T, 64]</c> replacing the silence src). <paramref name="inferSteps"/> overrides the turbo default 8 (upstream allows 1..20 on turbo via the same linspace+shift schedule; shift stays snapped to the trained {1,2,3}). Returns one <c>float[]</c> per channel at 48 kHz.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textHidden, Tensor? lyricHidden, double durationSeconds,
        float? shift = null, int? seed = null, Tensor? timbreLatent = null, Tensor? lmHints = null,
        int? inferSteps = null, Action<GenerationProgress>? onProgress = null)
        => Generate(textHidden, lyricHidden, durationSeconds,
            new AceStep15GenerateOptions { Shift = shift, Seed = seed, InferSteps = inferSteps },
            timbreLatent, lmHints, onProgress);

    /// <summary>Options-driven generation (CFG for base/sft, SDE, DCW). See <see cref="AceStep15GenerateOptions"/>.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textHidden, Tensor? lyricHidden, double durationSeconds, AceStep15GenerateOptions opts,
        Tensor? timbreLatent = null, Tensor? lmHints = null, Action<GenerationProgress>? onProgress = null)
        => GenerateCore(textHidden, lyricHidden, durationSeconds, opts, editPlan: null, timbreLatent, lmHints, onProgress, CancellationToken.None);

    /// <summary>Audio-conditioned generation: <paramref name="editPlan"/> supplies the <c>src_latents</c> slot, the per-frame chunk mask (1 = generate, 0 = preserve) and the schedule entry point, which together drive continuation, repaint and cover. Preserved frames are re-asserted from the source after every step (the standard repaint re-noise), so they survive rather than merely conditioning. A null plan is byte-identical to the plain text-to-music path. <b>Parity-pending: no part of this path is verified against real weights.</b> Two deliberate divergences: the reference preserves purely through the context (it always runs the full schedule from noise and never re-asserts the source), and the reference's cover strength blends a cover-instruction and a text2music-instruction velocity per step instead of moving the schedule entry point. Cover here also feeds raw 25 Hz Oobleck latents where upstream feeds FSQ-detokenized 5 Hz hints (that tokenizer is not ported). The mask polarity follows upstream's boolean <c>ones()</c> mask.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textHidden, Tensor? lyricHidden, double durationSeconds, AceStep15GenerateOptions opts,
        AceStep15EditPlan? editPlan, Tensor? timbreLatent = null, Tensor? lmHints = null,
        Action<GenerationProgress>? onProgress = null, CancellationToken cancel = default)
        => GenerateCore(textHidden, lyricHidden, durationSeconds, opts, editPlan, timbreLatent, lmHints, onProgress, cancel);

    private (float[] Left, float[] Right, int SampleRate, int Seed) GenerateCore(
        Tensor textHidden, Tensor? lyricHidden, double durationSeconds, AceStep15GenerateOptions opts,
        AceStep15EditPlan? editPlan, Tensor? timbreLatent, Tensor? lmHints,
        Action<GenerationProgress>? onProgress, CancellationToken cancel)
    {
        ThrowIfDisposed();
        if (durationSeconds < 1 || durationSeconds > 600)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "duration must be 1..600 s.");

        int actualSeed = opts.Seed ?? SeedGenerator.RandomSeed();
        int frames = _config.FrameCount(durationSeconds);
        // Turbo: 1..20 steps, shift snapped to trained {1,2,3}. Base/sft: arbitrary steps/shift.
        int wantSteps = _config.IsTurbo ? Math.Clamp(opts.InferSteps ?? _config.NumInferenceSteps, 1, 20)
            : Math.Clamp(opts.InferSteps ?? 50, 1, 200);
        float[] timesteps = AceStep15Config.GetTimesteps(wantSteps, opts.Shift ?? _config.FlowShift, snapShift: _config.IsTurbo);
        int steps = timesteps.Length;

        Tensor? srcLatents = lmHints;
        float[]? chunkMask = null;
        bool silenceMaskedFrames = false;
        int startIndex = 0;
        if (editPlan is not null)
        {
            if (lmHints is not null)
            {
                throw new ArgumentException("An edit plan and LM hints both occupy the src_latents slot — supply only one.", nameof(editPlan));
            }
            editPlan.Validate(frames, _config.LatentChannels);
            srcLatents = editPlan.SrcLatent;
            chunkMask = editPlan.ChunkMask;
            silenceMaskedFrames = editPlan.SilenceMaskedFrames;
            // Enter the descending table at the first sigma the plan allows; at least one step always runs.
            while (startIndex < steps - 1 && timesteps[startIndex] > editPlan.StartSigmaFraction)
            {
                startIndex++;
            }
        }

        // Turbo bakes guidance into distillation — upstream force-clamps to 1.0 (their issue #927: CFG on
        // turbo = noise/NaN). Base/sft need the learned null_condition_emb for the uncond branch.
        float guidance = opts.GuidanceScale;
        if (_config.IsTurbo && guidance != 1f)
        {
            Logs.Info($"ACE-Step 1.5: turbo model — overriding guidance {guidance:0.0} -> 1.0 (turbo does not use CFG).");
            guidance = 1f;
        }
        bool useCfg = guidance > 1f && _nullConditionEmb is not null;
        if (guidance > 1f && _nullConditionEmb is null)
        {
            Logs.Warning("ACE-Step 1.5: guidance requested but null_condition_emb not loaded — running uncond-free.");
        }
        bool sde = string.Equals(opts.InferMethod, "sde", StringComparison.OrdinalIgnoreCase);
        // "apg" | "cfg" | "adg"; empty honors the deprecated UseAdg bool.
        string guidanceType = string.IsNullOrEmpty(opts.GuidanceType) ? (opts.UseAdg ? "adg" : "apg")
            : opts.GuidanceType.ToLowerInvariant();
        if (guidanceType is not ("apg" or "cfg" or "adg"))
            throw new ArgumentException($"Unknown ACE-Step guidance type '{opts.GuidanceType}' — expected apg, cfg, or adg.");
        AceStep15Guidance.DcwCorrector dcw = new(opts.DcwEnabled ?? _config.IsTurbo, opts.DcwMode, opts.DcwScaler, opts.DcwHighScaler);
        AceStep15Guidance.MomentumBuffer momentum = new();

        Logs.Info($"ACE-Step 1.5 {(_config.IsTurbo ? "turbo" : "base/sft")}: {durationSeconds:0}s ({frames} latent frames), {steps} steps, " +
            $"shift={opts.Shift ?? _config.FlowShift}, cfg={(useCfg ? guidance : 1f)}{(useCfg ? $" ({guidanceType.ToUpperInvariant()})" : "")}, " +
            $"method={(sde ? "sde" : "ode")}, dcw={dcw.IsActive}, lyrics={(lyricHidden is null ? 0 : lyricHidden.Shape[lyricHidden.Shape.Rank - 2])} tokens, " +
            $"timbre={(timbreLatent is not null)}, hints={(lmHints is not null)}, seed={actualSeed}" +
            $"{(editPlan is null ? "" : $", edit=[start step {startIndex}/{steps}, sigma {timesteps[startIndex]:0.000}]")}");

        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor conditions = _encoder.EncodeConditions(Backend, textHidden, lyricHidden, timbreLatent);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());

        // Uncond conditions = null_condition_emb broadcast over the packed sequence (encoder NOT re-run).
        Tensor? nullConditions = null;
        if (useCfg)
        {
            int condLen = (int)conditions.Shape[1];
            int condDim = (int)conditions.Shape[2];
            nullConditions = new Tensor(new TensorShape(1, condLen, condDim), DType.F32);
            float* np = (float*)nullConditions.DataPointer;
            float* ne = (float*)_nullConditionEmb!.DataPointer;
            for (int i = 0; i < condLen; i++)
                Buffer.MemoryCopy(ne, np + (long)i * condDim, condDim * 4, condDim * 4);
        }

        Tensor context = BuildContextLatents(frames, srcLatents, chunkMask, silenceMaskedFrames);
        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, frames, _config.LatentChannels), actualSeed);
        long elems = z.Shape.ElementCount;
        if (editPlan is not null)
        {
            // Flow-matching forward noising at the entry sigma: z = (1 - t)·src + t·noise (t = 1 leaves pure noise).
            float sigmaStart = timesteps[startIndex];
            float* initZ = (float*)z.DataPointer;
            float* initSrc = (float*)editPlan.SrcLatent.DataPointer;
            for (long e = 0; e < elems; e++)
            {
                initZ[e] = (1f - sigmaStart) * initSrc[e] + sigmaStart * initZ[e];
            }
        }

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = startIndex; i < steps; i++)
            {
                cancel.ThrowIfCancellationRequested();
                float sigma = timesteps[i];
                float sigmaNext = i < steps - 1 ? timesteps[i + 1] : 0f;   // final step integrates to 0
                Tensor vt = _dit.Forward(Backend, z, context, conditions, sigma, sigma);   // r = t (all variants)

                if (useCfg)
                {
                    // Sequential dual batch-1 forward (the DiT is batch-1 throughout); interval-gated blend.
                    bool applyCfg = sigma >= opts.CfgIntervalStart && sigma <= opts.CfgIntervalEnd;
                    if (applyCfg)
                    {
                        Tensor vu = _dit.Forward(Backend, z, context, nullConditions!, sigma, sigma);
                        Tensor guided = guidanceType switch
                        {
                            "adg" => AceStep15Guidance.AdgForward(z, vt, vu, sigma, guidance),
                            "cfg" => AceStep15Guidance.CfgForward(vt, vu, guidance),
                            _ => AceStep15Guidance.ApgForward(vt, vu, guidance, momentum),
                        };
                        vu.Dispose();
                        vt.Dispose();
                        vt = guided;
                    }
                }

                float* zp = (float*)z.DataPointer;
                float* vp = (float*)vt.DataPointer;
                // DCW needs the clean estimate from the PRE-update latent: denoised = x_before − t_curr·v.
                Tensor? denoised = null;
                if (dcw.IsActive)
                {
                    denoised = new Tensor(z.Shape, DType.F32);
                    float* dp = (float*)denoised.DataPointer;
                    for (long e = 0; e < elems; e++) dp[e] = zp[e] - sigma * vp[e];
                }
                if (sde)
                {
                    // Predict clean (x − t·v), renoise at the LINEAR next timestep with a fresh draw.
                    float nextT = 1f - (i + 1) / (float)steps;
                    Tensor fresh = SeedGenerator.CreateNoise(z.Shape, unchecked(actualSeed + 7919 * (i + 1)));
                    float* fp = (float*)fresh.DataPointer;
                    for (long e = 0; e < elems; e++)
                    {
                        float clean = zp[e] - sigma * vp[e];
                        zp[e] = nextT * fp[e] + (1f - nextT) * clean;
                    }
                    fresh.Dispose();
                }
                else
                {
                    float dt = sigmaNext - sigma;
                    for (long e = 0; e < elems; e++) zp[e] += dt * vp[e];
                }
                if (denoised is not null)
                {
                    dcw.Apply(z, denoised, sigma);
                    denoised.Dispose();
                }
                if (chunkMask is not null)
                {
                    RestorePreservedFrames(z, editPlan!.SrcLatent, chunkMask, sigmaNext, unchecked(actualSeed + 7919 * (i + 1)));
                }
                vt.Dispose();
                onProgress?.Invoke(new GenerationProgress(i - startIndex + 1, steps - startIndex, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            conditions.Dispose();
            nullConditions?.Dispose();
            context.Dispose();
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());

        Tensor latentCf = new Tensor(new TensorShape(1, _config.LatentChannels, frames), DType.F32);
        Backend.Transpose2D(latentCf, z, frames, _config.LatentChannels);
        z.Dispose();
        Backend.PreloadWeights(_vae.EnumerateWeights());
        Tensor wav = _vae.Decode(Backend, latentCf);
        latentCf.Dispose();
        Backend.Sync();
        Backend.FreeWeights(_vae.EnumerateWeights());

        int channelSamples = (int)wav.Shape[2];
        float[] left = new float[channelSamples];
        float[] right = new float[channelSamples];
        float* wp = (float*)wav.DataPointer;
        new ReadOnlySpan<float>(wp, channelSamples).CopyTo(left);
        new ReadOnlySpan<float>(wp + (wav.Shape[1] > 1 ? channelSamples : 0), channelSamples).CopyTo(right);
        wav.Dispose();

        // Peak-normalize to -1 dBFS — upstream's enable_normalization default (audio_utils.normalize_audio).
        float peak = 0f;
        for (int i = 0; i < channelSamples; i++) peak = MathF.Max(peak, MathF.Max(MathF.Abs(left[i]), MathF.Abs(right[i])));
        if (peak > 1e-6f)
        {
            float gain = 0.8912509f / peak;   // 10^(-1/20)
            for (int i = 0; i < channelSamples; i++) { left[i] *= gain; right[i] *= gain; }
        }

        Logs.Info($"ACE-Step 1.5 complete: {channelSamples / (double)_config.SampleRate:0.0}s stereo, seed={actualSeed}");
        return (left, right, _config.SampleRate, actualSeed);
    }

    /// <summary>Builds <c>context_latents [1, T, 128]</c> = src ‖ chunk-mask per frame: src = <paramref name="srcLatents"/> (LM hints or an edit plan's source) when given, else the silence latent tiled; mask = <paramref name="chunkMask"/> when given, else 1.0 everywhere (upstream's full-generation bool ones() mask — its "auto" 2.0 assign clamps to True=1.0). With <paramref name="silenceMaskedFrames"/> the src row of every generate-masked frame becomes the tiled silence latent, which is upstream's repaint rule and also gives a continuation's appended tail a src row.</summary>
    private Tensor BuildContextLatents(int frames, Tensor? srcLatents, float[]? chunkMask, bool silenceMaskedFrames)
    {
        int latCh = _config.LatentChannels;
        if (srcLatents is not null && (srcLatents.Shape.Rank != 3 || srcLatents.Shape[1] != frames || (int)srcLatents.Shape[2] != latCh))
            throw new ArgumentException($"src latents must be [1, {frames}, {latCh}]; got {srcLatents.Shape}.", nameof(srcLatents));
        if (chunkMask is not null && chunkMask.Length != frames)
            throw new ArgumentException($"chunk mask must have {frames} entries; got {chunkMask.Length}.", nameof(chunkMask));
        bool maskedToSilence = silenceMaskedFrames && chunkMask is not null;
        if (srcLatents is null || maskedToSilence) EnsureSilenceFrames();

        Tensor context = new Tensor(new TensorShape(1, frames, 2 * latCh), DType.F32);
        float* cp = (float*)context.DataPointer;
        float* hp = srcLatents is not null ? (float*)srcLatents.DataPointer : null;
        float* sp = _silenceFrames is not null ? (float*)_silenceFrames.DataPointer : null;
        int silencePeriod = _silenceFrames is not null ? (int)_silenceFrames.Shape[0] : 1;
        for (int i = 0; i < frames; i++)
        {
            long rowOff = (long)i * 2 * latCh;
            float mask = chunkMask is null ? 1f : chunkMask[i];
            bool useSilence = srcLatents is null || (maskedToSilence && mask > 0.5f);
            float* row = useSilence ? sp + (long)(i % silencePeriod) * latCh : hp + (long)i * latCh;
            Buffer.MemoryCopy(row, cp + rowOff, latCh * 4, latCh * 4);
            for (int c = 0; c < latCh; c++) cp[rowOff + latCh + c] = mask;
        }
        return context;
    }

    /// <summary>Re-asserts the source over every masked-preserve frame (mask &lt; 0.5) after a denoise step: the standard repaint re-noise <c>z = (1 − t_next)·src + t_next·noise</c>, which collapses to writing the clean source back on the final step (t_next = 0). Without it a 0 mask only conditions the DiT instead of preserving the audio.</summary>
    private static void RestorePreservedFrames(Tensor z, Tensor src, float[] chunkMask, float sigmaNext, int noiseSeed)
    {
        int frames = chunkMask.Length;
        int latCh = (int)z.Shape[2];
        bool anyPreserved = false;
        for (int f = 0; f < frames && !anyPreserved; f++)
        {
            anyPreserved = chunkMask[f] < 0.5f;
        }
        if (!anyPreserved)
        {
            return;
        }
        Tensor? noise = sigmaNext > 0f ? SeedGenerator.CreateNoise(z.Shape, noiseSeed) : null;
        float* zp = (float*)z.DataPointer;
        float* sp = (float*)src.DataPointer;
        float* np = noise is not null ? (float*)noise.DataPointer : null;
        for (int f = 0; f < frames; f++)
        {
            if (chunkMask[f] >= 0.5f)
            {
                continue;
            }
            long off = (long)f * latCh;
            for (int c = 0; c < latCh; c++)
            {
                long e = off + c;
                zp[e] = np is null ? sp[e] : (1f - sigmaNext) * sp[e] + sigmaNext * np[e];
            }
        }
        noise?.Dispose();
    }

    /// <summary>Computes the silence src latent once: 1 s of digital silence through the VAE encoder (the research doc's sanctioned recompute of the shipped <c>silence_latent.pt</c>). Falls back to zero latents when the injected decoder cannot encode — validation-gated: parity with the .pt asset is unverified either way until a real-checkpoint dump comparison.</summary>
    private void EnsureSilenceFrames()
    {
        if (_silenceFrames is not null) return;
        int latCh = _config.LatentChannels;
        if (_vae is IAudioLatentEncoder encoder && encoder.CanEncode)
        {
            int samples = _config.LatentRate * _config.SamplesPerLatent;
            Tensor pcm = new Tensor(new TensorShape(1, encoder.AudioChannels, samples), DType.F32);
            new Span<float>((float*)pcm.DataPointer, (int)pcm.Shape.ElementCount).Clear();
            Backend.PreloadWeights(_vae.EnumerateWeights());
            Tensor latent = encoder.EncodeMode(Backend, pcm);
            Backend.Sync();
            Backend.FreeWeights(_vae.EnumerateWeights());
            pcm.Dispose();
            int tLat = (int)latent.Shape[2];
            Tensor rows = new Tensor(new TensorShape(tLat, latCh), DType.F32);
            Backend.Transpose2D(rows, latent, latCh, tLat);
            latent.Dispose();
            _silenceFrames = rows;
        }
        else
        {
            Logs.Warning("ACE-Step 1.5: VAE cannot encode — using zero src latents instead of the silence latent.");
            Tensor rows = new Tensor(new TensorShape(1, latCh), DType.F32);
            new Span<float>((float*)rows.DataPointer, latCh).Clear();
            _silenceFrames = rows;
        }
    }

    protected override void DisposeCore()
    {
        _silenceFrames?.Dispose();
        _silenceFrames = null;
        _nullConditionEmb?.Dispose();
        _nullConditionEmb = null;
    }
}
