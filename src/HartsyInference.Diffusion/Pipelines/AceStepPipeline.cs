using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>ACE-Step v1 (ACE Studio + StepFun, Apache-2.0) text/lyrics-to-music pipeline — the 3.5B flow-matching DiT
/// over Music-DCAE mel latents: pre-computed UMT5-base features + VoiceBpe lyric ids → tri-source cross-attention
/// context → shift-3 flow-match denoise (Euler / Heun / PingPong) with CFG / APG / CFG-Zero★ guidance → DCAE decode
/// to stereo log-mel → ADaMoS HiFi-GAN per channel → 44.1 kHz stereo waveform. Token rate ≈ 10.77 Hz (a 4-minute
/// song ≈ 2585 DiT tokens). Encode UMT5 upstream with the shared <c>T5TextEncoder</c>
/// (<c>T5TextEncoderConfig.Umt5Base</c>) and tokenize lyrics with <c>AceStepLyricTokenizer</c>.
/// <b>Status: built, first-run validation pending</b> — guidance defaults (CFG 7.0 / 27 steps) per the model card;
/// APG momentum drifts from the Python reference by design (validation uses plain CFG).
/// <para><b>CUDA (3060/12 GB):</b> runs end-to-end with bf16 weights resident (F32 won't fit), but the bf16 GEMM
/// truncates F32 activations and the latent NaNs over the CFG loop. The caller MUST set
/// <c>CudaBackend.HighPrecisionGemm = true</c> before constructing this pipeline — it upcasts bf16 weights to F32 for
/// the GEMM (weights stay bf16-resident) and yields finite audio (verified: 4 s / 8 steps ≈ 27 s on a 3060). The
/// Diffusion package can't reference the Cuda backend, so this is wired at the call site (SwarmUI loader / tests).</para></summary>
public sealed unsafe class AceStepPipeline : DiffusionPipelineBase
{
    /// <summary>Guidance modes (research § 2.9).</summary>
    public enum GuidanceMode
    {
        Cfg,
        Apg,
        CfgZeroStar,
    }

    /// <summary>Sampler choices (all share the shift-3 sigma grid).</summary>
    public enum SamplerMode
    {
        Euler,
        Heun,
        PingPong,
    }

    private readonly AceStepDit _dit;
    private readonly MusicDcaeDecoder _dcae;
    private readonly AdaMosHiFiGanV1 _vocoder;
    private readonly AceStepConfig _config;

    public AceStepPipeline(IBackend backend, AceStepDit dit, MusicDcaeDecoder dcae, AdaMosHiFiGanV1 vocoder,
        AceStepConfig config)
        : base(backend)
    {
        _dit = dit;
        _dcae = dcae;
        _vocoder = vocoder;
        _config = config;
    }

    /// <summary>Generates stereo audio from pre-computed UMT5 style features <c>[T_text, 768]</c> and ACE lyric token
    /// ids (empty array → instrumental via prompt tags). Returns one <c>float[]</c> per channel at 44.1 kHz.
    /// <para>The full upstream <c>pipeline_ace_step.py</c> guidance controls are exposed per-call (all default to the
    /// config): <paramref name="guidanceInterval"/> (0.5, CFG only over the centered trajectory fraction),
    /// <paramref name="guidanceIntervalDecay"/> (0.0) toward <paramref name="minGuidanceScale"/> (3.0),
    /// <paramref name="omegaScale"/> (10.0, the scheduler granularity rescale), double-condition CFG
    /// (<paramref name="guidanceScaleText"/> / <paramref name="guidanceScaleLyric"/>, both must exceed 1.0 to enable),
    /// and <paramref name="ossSteps"/> (an optimal custom-step subset; 1-indexed into the full grid).</para></summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textEmbeds, int[] lyricIds, double durationSeconds,
        int? steps = null, float? guidance = null, GuidanceMode guidanceMode = GuidanceMode.Apg,
        SamplerMode sampler = SamplerMode.Euler, int? seed = null, float[]? speakerVec = null,
        float? guidanceInterval = null, float? guidanceIntervalDecay = null, float? minGuidanceScale = null,
        float? omegaScale = null, float? guidanceScaleText = null, float? guidanceScaleLyric = null,
        int[]? ossSteps = null, Action<GenerationProgress>? onProgress = null,
        Tensor? ergTextEmbeds = null, bool useErgLyric = false, bool useErgDiffusion = false)
    {
        ThrowIfDisposed();
        if (durationSeconds < 1 || durationSeconds > 600)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "duration must be 1..600 s.");

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int inferSteps = steps ?? _config.NumInferenceSteps;
        float g = guidance ?? _config.GuidanceScale;
        float gInterval = guidanceInterval ?? _config.GuidanceInterval;
        float gIntervalDecay = guidanceIntervalDecay ?? _config.GuidanceIntervalDecay;
        float minG = minGuidanceScale ?? _config.MinGuidanceScale;
        float omega = omegaScale ?? _config.OmegaScale;
        float gText = guidanceScaleText ?? _config.GuidanceScaleText;
        float gLyric = guidanceScaleLyric ?? _config.GuidanceScaleLyric;
        int fLat = Math.Max(8, _config.LatentFrames(durationSeconds));

        // Custom (OSS) step schedule: effective steps = |oss|, each sampling a sigma from the full grid (1-indexed).
        float[] sigmas = ossSteps is { Length: > 0 }
            ? BuildOssSigmas(ossSteps, _config.FlowShift)
            : BuildSigmas(inferSteps, _config.FlowShift);
        int loopSteps = sigmas.Length - 1;

        // Guidance is applied only over the centered interval [start, end) (upstream guidance_interval).
        bool doCfg = g != 0f && g != 1f;
        bool doDualCondition = gText > 1f && gLyric > 1f;
        int startIdx = (int)(loopSteps * ((1f - gInterval) / 2f));
        int endIdx = (int)(loopSteps * (gInterval / 2f + 0.5f));
        // omega mean-preserving rescale factor (constant across steps).
        float omegaNorm = 0.9f + 0.2f / (1f + MathF.Exp(-0.1f * omega));

        Logs.Info($"ACE-Step: {durationSeconds:0}s ({fLat} latent frames), {loopSteps} steps, {guidanceMode} g={g}, " +
            $"interval [{startIdx},{endIdx}) decay={gIntervalDecay} minG={minG} omega={omega}, " +
            (doDualCondition ? $"dualCFG(text={gText},lyric={gLyric}), " : "") +
            $"{sampler}, {lyricIds.Length} lyric tokens, seed={actualSeed}");

        Backend.PreloadWeights(_dit.EnumerateWeights());

        Tensor ctx = _dit.BuildContext(Backend, textEmbeds, lyricIds, speakerVec);
        // ERG (upstream use_erg_lyric, default ON there): the uncond context is a real encode with weakened
        // text (use_erg_tag's τ-scaled UMT5 pass, else zeros) and the lyric encoder's Q τ-scaled in layers
        // 4..6 — NOT the zeroed context of the plain-CFG branch.
        Tensor uncondCtx;
        if (useErgLyric)
        {
            Tensor nullText = ergTextEmbeds ?? new Tensor(textEmbeds.Shape, DType.F32);
            uncondCtx = _dit.BuildContext(Backend, nullText, lyricIds, null, lyricQScale: (4, 6, 0.01f));
            if (!ReferenceEquals(nullText, ergTextEmbeds)) nullText.Dispose();
        }
        else
        {
            uncondCtx = new Tensor(ctx.Shape, DType.F32);   // zeroed conditioning (upstream non-ERG branch)
        }
        // Double-condition needs a "text-only" branch: real text + weakened lyric under ERG, else the full
        // context with the lyric rows zeroed.
        Tensor? onlyTextCtx = !doDualCondition ? null
            : useErgLyric
                ? _dit.BuildContext(Backend, textEmbeds, lyricIds, null, lyricQScale: (4, 6, 0.01f))
                : BuildOnlyTextContext(ctx, (int)textEmbeds.Shape[0]);

        Tensor z = SeedGenerator.CreateNoise(new TensorShape([1L, _config.InChannels, _config.LatentHeight, fLat]), actualSeed);
        using AceStepGuidance.MomentumBuffer momentum = new();

        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < loopSteps; i++)
            {
                float t = sigmas[i] * 1000f;
                bool inInterval = doCfg && i >= startIdx && i < endIdx;
                float curG = g;
                if (inInterval && gIntervalDecay > 0f)
                {
                    float progress = (i - startIdx) / (float)Math.Max(1, endIdx - startIdx - 1);
                    curG = g - (g - minG) * progress * gIntervalDecay;
                }
                Tensor v = inInterval
                    ? GuidedVelocity(z, ctx, uncondCtx, onlyTextCtx, t, curG, gText, gLyric, doDualCondition,
                        guidanceMode, momentum, zeroInit: i == startIdx, useErgDiffusion)
                    : _dit.Forward(Backend, z, ctx, t);
                float dt = sigmas[i + 1] - sigmas[i];
                switch (sampler)
                {
                    case SamplerMode.Euler:
                        EulerStep(z, v, dt, omegaNorm);
                        break;
                    case SamplerMode.Heun when i < loopSteps - 1:
                    {
                        Tensor mid = Clone(z);
                        EulerStep(mid, v, dt, omegaNorm);
                        bool nextInInterval = doCfg && (i + 1) >= startIdx && (i + 1) < endIdx;
                        Tensor v2 = nextInInterval
                            ? GuidedVelocity(mid, ctx, uncondCtx, onlyTextCtx, sigmas[i + 1] * 1000f, curG, gText, gLyric,
                                doDualCondition, guidanceMode, momentum, zeroInit: false, useErgDiffusion)
                            : _dit.Forward(Backend, mid, ctx, sigmas[i + 1] * 1000f);
                        mid.Dispose();
                        float* vp = (float*)v.DataPointer;
                        float* v2p = (float*)v2.DataPointer;
                        for (long e = 0; e < v.Shape.ElementCount; e++) vp[e] = 0.5f * (vp[e] + v2p[e]);
                        v2.Dispose();
                        EulerStep(z, v, dt, omegaNorm);
                        break;
                    }
                    case SamplerMode.Heun:
                        EulerStep(z, v, dt, omegaNorm);   // final step degrades to Euler
                        break;
                    case SamplerMode.PingPong:
                    {
                        float sigma = sigmas[i], sigmaNext = sigmas[i + 1];
                        Tensor fresh = SeedGenerator.CreateNoise(z.Shape, actualSeed + i + 1);
                        float* zp = (float*)z.DataPointer;
                        float* vp = (float*)v.DataPointer;
                        float* np = (float*)fresh.DataPointer;
                        for (long e = 0; e < z.Shape.ElementCount; e++)
                        {
                            float x0 = zp[e] - sigma * vp[e];
                            zp[e] = (1f - sigmaNext) * x0 + sigmaNext * np[e];
                        }
                        fresh.Dispose();
                        break;
                    }
                }
                v.Dispose();
                onProgress?.Invoke(new GenerationProgress(i + 1, loopSteps, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            ctx.Dispose();
            uncondCtx.Dispose();
            onlyTextCtx?.Dispose();
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());

        // Invert the pipeline-level latent scaling (NOT the diffusers 0.41407 — research § 7).
        float* lp = (float*)z.DataPointer;
        for (long e = 0; e < z.Shape.ElementCount; e++)
            lp[e] = lp[e] / _config.LatentScaleFactor + _config.LatentShiftFactor;

        Tensor mel = _dcae.Decode(Backend, z);
        z.Dispose();
        // De-standardize ([-1,1] → [0,1]) then back to the clipped log-mel range [-11, +3].
        float* mp = (float*)mel.DataPointer;
        for (long e = 0; e < mel.Shape.ElementCount; e++)
            mp[e] = (mp[e] * 0.5f + 0.5f) * 14f - 11f;

        Backend.PreloadWeights(_vocoder.EnumerateWeights());
        float[] left = DecodeChannel(mel, 0);
        float[] right = DecodeChannel(mel, 1);
        mel.Dispose();
        Backend.FreeWeights(_vocoder.EnumerateWeights());

        Logs.Info($"ACE-Step complete: {left.Length / (double)_config.SampleRate:0.0}s stereo, seed={actualSeed}");
        return (left, right, _config.SampleRate, actualSeed);
    }

    private Tensor GuidedVelocity(Tensor z, Tensor ctx, Tensor uncondCtx, Tensor? onlyTextCtx, float t, float g,
        float gText, float gLyric, bool doDualCondition, GuidanceMode mode,
        AceStepGuidance.MomentumBuffer momentum, bool zeroInit, bool useErgDiffusion = false)
    {
        // ERG diffusion (upstream use_erg_diffusion): the UNCOND forwards run with blocks 15..20's self+cross
        // attention Q τ-scaled; the cond forward is untouched.
        (int, int, float)? uncondQScale = useErgDiffusion ? (15, 20, 0.01f) : null;
        Tensor vCond = _dit.Forward(Backend, z, ctx, t);

        // Double-condition CFG: uncond → text-only → full, blended (upstream cfg_double_condition_forward).
        if (doDualCondition && onlyTextCtx is not null)
        {
            Tensor vText = _dit.Forward(Backend, z, onlyTextCtx, t, uncondQScale);
            Tensor vUncondD = _dit.Forward(Backend, z, uncondCtx, t, uncondQScale);
            float* c = (float*)vCond.DataPointer;
            float* tx = (float*)vText.DataPointer;
            float* un = (float*)vUncondD.DataPointer;
            for (long e = 0; e < vCond.Shape.ElementCount; e++)
                c[e] = (1f - gText) * un[e] + (gText - gLyric) * tx[e] + gLyric * c[e];
            vText.Dispose();
            vUncondD.Dispose();
            return vCond;
        }

        Tensor vUncond = _dit.Forward(Backend, z, uncondCtx, t, uncondQScale);
        switch (mode)
        {
            case GuidanceMode.Cfg:
                AceStepGuidance.Cfg(vCond, vUncond, g);
                break;
            case GuidanceMode.Apg:
                AceStepGuidance.Apg(vCond, vUncond, g, momentum);
                break;
            case GuidanceMode.CfgZeroStar:
                AceStepGuidance.CfgZeroStar(vCond, vUncond, g, zeroInit);
                break;
        }
        vUncond.Dispose();
        return vCond;
    }

    /// <summary>Clones the full context and zeroes the lyric rows (everything after speaker + text), leaving the
    /// speaker + UMT5 text conditioning active — the "text-only" branch for double-condition CFG.</summary>
    private Tensor BuildOnlyTextContext(Tensor ctx, int textRows)
    {
        Tensor o = Clone(ctx);
        int dim = (int)ctx.Shape[1];
        int lyricStart = 1 + textRows;   // row 0 = speaker, rows 1..textRows = text.
        long rows = ctx.Shape[0];
        if (lyricStart < rows)
        {
            float* p = (float*)o.DataPointer;
            new Span<float>(p + (long)lyricStart * dim, (int)((rows - lyricStart) * dim)).Clear();
        }
        return o;
    }

    private float[] DecodeChannel(Tensor mel, int channel)
    {
        int bins = (int)mel.Shape[2], frames = (int)mel.Shape[3];
        Tensor mono = new Tensor(new TensorShape(1, bins, frames), DType.F32);
        float* src = (float*)mel.DataPointer + (long)channel * bins * frames;
        Buffer.MemoryCopy(src, (float*)mono.DataPointer, (long)bins * frames * 4, (long)bins * frames * 4);
        Tensor wav = _vocoder.Decode(Backend, mono);
        mono.Dispose();
        float[] samples = new float[wav.Shape.ElementCount];
        new ReadOnlySpan<float>((float*)wav.DataPointer, samples.Length).CopyTo(samples);
        wav.Dispose();
        return samples;
    }

    private static float[] BuildSigmas(int steps, float shift)
    {
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float t = 1.0f - (float)i / steps;
            sigmas[i] = shift * t / (1.0f + (shift - 1.0f) * t);
        }
        return sigmas;
    }

    /// <summary>Custom (OSS) sigma schedule: build the full grid of <c>max(ossSteps)</c> steps, then keep the sigmas
    /// at the (1-indexed) <paramref name="ossSteps"/> positions plus a terminal 0 (upstream <c>oss_steps</c>).</summary>
    private static float[] BuildOssSigmas(int[] ossSteps, float shift)
    {
        int maxStep = 0;
        foreach (int s in ossSteps) maxStep = Math.Max(maxStep, s);
        float[] full = BuildSigmas(maxStep, shift);
        float[] sel = new float[ossSteps.Length + 1];
        for (int j = 0; j < ossSteps.Length; j++)
            sel[j] = full[Math.Clamp(ossSteps[j] - 1, 0, maxStep)];
        sel[ossSteps.Length] = 0f;
        return sel;
    }

    /// <summary>Flow-match Euler step with ACE-Step's omega granularity rescale: the mean of the Euler delta is
    /// preserved while its residual is scaled by <paramref name="omegaNorm"/> (upstream <c>omega_scale</c> effect).</summary>
    private static void EulerStep(Tensor target, Tensor value, float dt, float omegaNorm)
    {
        float* tp = (float*)target.DataPointer;
        float* vp = (float*)value.DataPointer;
        long n = target.Shape.ElementCount;
        if (omegaNorm == 1f)
        {
            for (long e = 0; e < n; e++) tp[e] += dt * vp[e];
            return;
        }
        double sum = 0;
        for (long e = 0; e < n; e++) sum += (double)dt * vp[e];
        float mean = (float)(sum / n);
        for (long e = 0; e < n; e++)
        {
            float dx = dt * vp[e];
            tp[e] += (dx - mean) * omegaNorm + mean;
        }
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
        return o;
    }
}
