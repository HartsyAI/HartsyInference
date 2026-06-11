using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Music;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>ACE-Step v1 (ACE Studio + StepFun, Apache-2.0) text/lyrics-to-music pipeline — the 3.5B flow-matching DiT
/// over Music-DCAE mel latents: pre-computed UMT5-base features + VoiceBpe lyric ids → tri-source cross-attention
/// context → shift-3 flow-match denoise (Euler / Heun / PingPong) with CFG / APG / CFG-Zero★ guidance → DCAE decode
/// to stereo log-mel → ADaMoS HiFi-GAN per channel → 44.1 kHz stereo waveform. Token rate ≈ 10.77 Hz (a 4-minute
/// song ≈ 2585 DiT tokens). Encode UMT5 upstream with the shared <c>T5TextEncoder</c>
/// (<c>T5TextEncoderConfig.Umt5Base</c>) and tokenize lyrics with <c>AceStepLyricTokenizer</c>.
/// <b>Status: built, first-run validation pending</b> — guidance defaults (CFG 7.0 / 27 steps) per the model card;
/// APG momentum drifts from the Python reference by design (validation uses plain CFG).</summary>
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
    /// ids (empty array → instrumental via prompt tags). Returns one <c>float[]</c> per channel at 44.1 kHz.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textEmbeds, int[] lyricIds, double durationSeconds,
        int? steps = null, float? guidance = null, GuidanceMode guidanceMode = GuidanceMode.Apg,
        SamplerMode sampler = SamplerMode.Euler, int? seed = null, float[]? speakerVec = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (durationSeconds < 1 || durationSeconds > 600)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "duration must be 1..600 s.");

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int inferSteps = steps ?? _config.NumInferenceSteps;
        float g = guidance ?? _config.GuidanceScale;
        int fLat = Math.Max(8, _config.LatentFrames(durationSeconds));

        Logs.Info($"ACE-Step: {durationSeconds:0}s ({fLat} latent frames), {inferSteps} steps, {guidanceMode} g={g}, " +
            $"{sampler}, {lyricIds.Length} lyric tokens, seed={actualSeed}");
        Logs.Warning("ACE-Step pipeline is first-run-validation pending — numerics unverified vs the reference.");

        Backend.PreloadWeights(_dit.EnumerateWeights());

        Tensor ctx = _dit.BuildContext(Backend, textEmbeds, lyricIds, speakerVec);
        Tensor uncondCtx = new Tensor(ctx.Shape, DType.F32);   // zeroed conditioning (matches the reference)

        float[] sigmas = BuildSigmas(inferSteps, _config.FlowShift);
        Tensor z = SeedGenerator.CreateNoise(new TensorShape([1L, _config.InChannels, _config.LatentHeight, fLat]), actualSeed);
        using AceStepGuidance.MomentumBuffer momentum = new();

        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < inferSteps; i++)
            {
                float t = sigmas[i] * 1000f;
                Tensor v = GuidedVelocity(z, ctx, uncondCtx, t, g, guidanceMode, momentum, zeroInit: i == 0);
                float dt = sigmas[i + 1] - sigmas[i];
                switch (sampler)
                {
                    case SamplerMode.Euler:
                        AddScaled(z, v, dt);
                        break;
                    case SamplerMode.Heun when i < inferSteps - 1:
                    {
                        Tensor mid = Clone(z);
                        AddScaled(mid, v, dt);
                        Tensor v2 = GuidedVelocity(mid, ctx, uncondCtx, sigmas[i + 1] * 1000f, g, guidanceMode, momentum, zeroInit: false);
                        mid.Dispose();
                        float* vp = (float*)v.DataPointer;
                        float* v2p = (float*)v2.DataPointer;
                        for (long e = 0; e < v.Shape.ElementCount; e++) vp[e] = 0.5f * (vp[e] + v2p[e]);
                        v2.Dispose();
                        AddScaled(z, v, dt);
                        break;
                    }
                    case SamplerMode.Heun:
                        AddScaled(z, v, dt);   // final step degrades to Euler
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
                onProgress?.Invoke(new GenerationProgress(i + 1, inferSteps, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            ctx.Dispose();
            uncondCtx.Dispose();
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

    private Tensor GuidedVelocity(Tensor z, Tensor ctx, Tensor uncondCtx, float t, float g,
        GuidanceMode mode, AceStepGuidance.MomentumBuffer momentum, bool zeroInit)
    {
        Tensor vCond = _dit.Forward(Backend, z, ctx, t);
        Tensor vUncond = _dit.Forward(Backend, z, uncondCtx, t);
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

    private static void AddScaled(Tensor target, Tensor value, float scale)
    {
        float* tp = (float*)target.DataPointer;
        float* vp = (float*)value.DataPointer;
        for (long e = 0; e < target.Shape.ElementCount; e++) tp[e] += scale * vp[e];
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
        return o;
    }
}
