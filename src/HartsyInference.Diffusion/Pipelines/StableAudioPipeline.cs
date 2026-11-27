using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Stable Audio Open Small text-to-audio/SFX pipeline: pre-computed T5-base prompt states + a <c>seconds_total</c> timing token → <see cref="StableAudioDit"/> rectified-flow denoise (8-step <see cref="StableAudioPingPongScheduler"/>, no CFG — ARC-distilled) → Oobleck VAE decode → 44.1 kHz stereo, trimmed to the requested duration. The DiT always denoises the full trained-length latent (<see cref="StableAudioDitConfig.MaxLatentTokens"/>, ~11.89 s) regardless of the requested <c>secondsTotal</c> — duration is controlled purely through the timing conditioning token and a post-decode trim, matching upstream <c>generate_diffusion_cond</c>. <b>Status: DiT / VAE / timing conditioner components individually real-weight-verified (cosine 1.0 each, see PARITY_VERIFICATION.md); this composition is structurally wired but not yet end-to-end validated against a Python reference.</b></summary>
public sealed unsafe class StableAudioPipeline : DiffusionPipelineBase
{
    private readonly StableAudioDit _dit;
    private readonly IAudioLatentDecoder _vae;
    private readonly StableAudioNumberEmbedder _timingConditioner;
    private readonly StableAudioNumberEmbedder? _startConditioner;
    private readonly StableAudioDitConfig _config;

    /// <param name="timingConditioner">The <c>seconds_total</c> conditioner.</param>
    /// <param name="startConditioner">The <c>seconds_start</c> conditioner; required by <see cref="GenerateVPrediction"/>
    /// (Open 1.0), absent on Open Small.</param>
    public StableAudioPipeline(IBackend backend, StableAudioDit dit, IAudioLatentDecoder vae,
        StableAudioNumberEmbedder timingConditioner, StableAudioDitConfig config, StableAudioNumberEmbedder? startConditioner = null)
        : base(backend)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _timingConditioner = timingConditioner ?? throw new ArgumentNullException(nameof(timingConditioner));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _startConditioner = startConditioner;
    }

    /// <summary>Generates stereo audio from pre-computed T5-base prompt states <c>[1, T_text, 768]</c> (<c>T5TextEncoderConfig.T5Base</c>, max_length 64) and a requested duration. Returns one <c>float[]</c> per channel at 44.1 kHz, trimmed to <paramref name="secondsTotal"/> (clamped to the model's trained window).</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textEmbeds, double secondsTotal,
        int? steps = null, int? seed = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        double maxDurationSeconds = _config.MaxLatentTokens * (double)_config.VaeDownsample / _config.SampleRate;
        double duration = Math.Clamp(secondsTotal, 0.0, maxDurationSeconds);
        if (duration != secondsTotal)
            Logs.Warning($"Stable Audio: requested {secondsTotal:0.00}s clamped to the trained window ({maxDurationSeconds:0.00}s).");

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int inferSteps = steps ?? 8;
        int t = _config.MaxLatentTokens;

        Logs.Info($"Stable Audio: {duration:0.00}s ({t} latent frames), {inferSteps} pingpong steps, seed={actualSeed}");

        Backend.PreloadWeights(_timingConditioner.EnumerateWeights());
        Tensor timingToken = _timingConditioner.Embed(Backend, (float)duration);
        Backend.FreeWeights(_timingConditioner.EnumerateWeights());

        int lt = (int)textEmbeds.Shape[1];
        Tensor condTokens = new(new TensorShape(1, lt + 1, _config.CondTokenDim), DType.F32);
        Backend.Concat(condTokens, [textEmbeds, timingToken], dim: 1);

        StableAudioPingPongScheduler scheduler = new();
        scheduler.SetTimesteps(inferSteps);

        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, _config.IoChannels, t), actualSeed);

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < inferSteps; i++)
            {
                Tensor v = _dit.Forward(Backend, z, condTokens, timingToken, scheduler.Sigmas[i]);
                Tensor fresh = SeedGenerator.CreateNoise(z.Shape, actualSeed + i + 1);
                Tensor zNext = scheduler.Step(Backend, z, v, fresh, i);
                v.Dispose();
                fresh.Dispose();
                z.Dispose();
                z = zNext;
                onProgress?.Invoke(new GenerationProgress(i + 1, inferSteps, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            condTokens.Dispose();
            timingToken.Dispose();
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());

        Backend.PreloadWeights(_vae.EnumerateWeights());
        Tensor pcm = _vae.Decode(Backend, z);
        z.Dispose();
        Backend.FreeWeights(_vae.EnumerateWeights());

        int totalSamples = (int)pcm.Shape[2];
        int keepSamples = Math.Min(totalSamples, (int)(duration * _config.SampleRate));
        float[] left = TrimChannel(pcm, 0, keepSamples);
        float[] right = TrimChannel(pcm, 1, keepSamples);
        pcm.Dispose();

        Logs.Info($"Stable Audio complete: {left.Length / (double)_config.SampleRate:0.00}s stereo, seed={actualSeed}");
        return (left, right, _config.SampleRate, actualSeed);
    }

    /// <summary>Stable Audio Open 1.0 (<see cref="StableAudioDitConfig.Open1_0"/>): v-prediction EDM with classifier-free
    /// guidance, stepped by DPM++ 3M SDE over <see cref="StableAudioEdm.Sigmas"/>, as stable-audio-tools'
    /// <c>generate_diffusion_cond</c> runs the model card's settings. Cross-attention reads the prompt states followed by
    /// the <c>seconds_start</c> and <c>seconds_total</c> tokens; the global token is the two timing embeddings side by
    /// side. The unconditional branch is the official null embedding: zeros of the same length, with the same global
    /// token. <paramref name="textEmbeds"/> carries only the prompt's real tokens (no padding), which equals the masked
    /// padded encoding the official conditioner makes. Returns 44.1 kHz stereo trimmed to the requested window.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) GenerateVPrediction(
        Tensor textEmbeds, double secondsStart, double secondsTotal, int steps, float cfgScale, int? seed = null,
        float sigmaMin = StableAudioEdm.DefaultSigmaMin, float sigmaMax = StableAudioEdm.DefaultSigmaMax,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(textEmbeds);
        if (!_config.HasSecondsStart || _startConditioner is null)
            throw new InvalidOperationException("GenerateVPrediction needs a seconds_start conditioner (Stable Audio Open 1.0).");
        double maxDurationSeconds = _config.MaxLatentTokens * (double)_config.VaeDownsample / _config.SampleRate;
        double start = Math.Clamp(secondsStart, 0.0, maxDurationSeconds);
        double duration = Math.Clamp(secondsTotal, 0.0, maxDurationSeconds);
        if (duration != secondsTotal)
            Logs.Warning($"Stable Audio: requested {secondsTotal:0.00}s clamped to the trained window ({maxDurationSeconds:0.00}s).");
        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        bool useCfg = cfgScale != 1.0f;
        Logs.Info($"Stable Audio Open 1.0: {duration:0.00}s from {start:0.00}s, {steps} dpmpp_3m_sde steps, cfg {cfgScale}, seed={actualSeed}");

        Backend.PreloadWeights(_timingConditioner.EnumerateWeights());
        Backend.PreloadWeights(_startConditioner.EnumerateWeights());
        Tensor startToken = _startConditioner.Embed(Backend, (float)start);
        Tensor totalToken = _timingConditioner.Embed(Backend, (float)duration);
        Backend.FreeWeights(_startConditioner.EnumerateWeights());
        Backend.FreeWeights(_timingConditioner.EnumerateWeights());

        int lt = (int)textEmbeds.Shape[1], condDim = _config.CondTokenDim;
        Tensor condTokens = new(new TensorShape(1, lt + 2, condDim), DType.F32);
        Backend.Concat(condTokens, [textEmbeds, startToken, totalToken], dim: 1);
        Tensor globalCond = new(new TensorShape(1, 1, 2 * condDim), DType.F32);
        Backend.Concat(globalCond, [startToken, totalToken], dim: 2);
        startToken.Dispose();
        totalToken.Dispose();
        Tensor? nullTokens = null;
        if (useCfg)
        {
            nullTokens = new(new TensorShape(1, lt + 2, condDim), DType.F32);
            Backend.Scale(nullTokens, condTokens, 0f);
        }

        float[] sigmas = StableAudioEdm.Sigmas(steps, sigmaMin, sigmaMax);
        DpmPlusPlus3MSdeSampler sampler = new(sigmas, actualSeed);
        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, _config.IoChannels, _config.MaxLatentTokens), actualSeed);
        Backend.Scale(z, z, sigmas[0]);
        sampler.Reset(z.Shape);
        DelegateDenoisePredictor predictor = new(PredictionType.VPrediction, (x, sigma, _) =>
        {
            Tensor scaled = new(x.Shape, DType.F32);
            Backend.Scale(scaled, x, StableAudioEdm.InputScale(sigma));
            float t = StableAudioEdm.TimestepOf(sigma);
            try
            {
                Tensor cond = _dit.Forward(Backend, scaled, condTokens, globalCond, t);
                if (!useCfg)
                    return new DenoisePrediction(cond, cond);
                Tensor uncond = _dit.Forward(Backend, scaled, nullTokens!, globalCond, t);
                return new DenoisePrediction(cond, uncond, cfgScale);
            }
            finally
            {
                scaled.Dispose();
            }
        });

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < steps; i++)
            {
                sampler.Step(Backend, z, predictor, i);
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            // Reset releases the sampler's history, which it pins on the backend, and its Brownian noise source; a
            // sampler abandoned without it keeps those allocations until the backend is torn down.
            sampler.Reset(z.Shape);
            condTokens.Dispose();
            globalCond.Dispose();
            nullTokens?.Dispose();
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        (float[] left, float[] right) = DecodeAndTrim(z, start, duration);
        Logs.Info($"Stable Audio Open 1.0 complete: {left.Length / (double)_config.SampleRate:0.00}s stereo, seed={actualSeed}");
        return (left, right, _config.SampleRate, actualSeed);
    }

    /// <summary>Decodes the latent and keeps <c>[start, start + duration)</c> of each channel. Disposes <paramref name="z"/>.</summary>
    private (float[] Left, float[] Right) DecodeAndTrim(Tensor z, double start, double duration)
    {
        Backend.PreloadWeights(_vae.EnumerateWeights());
        Tensor pcm = _vae.Decode(Backend, z);
        z.Dispose();
        Backend.FreeWeights(_vae.EnumerateWeights());
        int totalSamples = (int)pcm.Shape[2];
        int first = Math.Min(totalSamples, (int)(start * _config.SampleRate));
        int keep = Math.Min(totalSamples - first, (int)(duration * _config.SampleRate));
        float[] left = SliceChannel(pcm, 0, first, keep);
        float[] right = SliceChannel(pcm, 1, first, keep);
        pcm.Dispose();
        return (left, right);
    }

    private static float[] SliceChannel(Tensor pcm, int channel, int first, int keep)
    {
        int totalSamples = (int)pcm.Shape[2];
        float[] samples = new float[keep];
        float* src = (float*)pcm.DataPointer + (long)channel * totalSamples + first;
        new ReadOnlySpan<float>(src, keep).CopyTo(samples);
        return samples;
    }

    private static float[] TrimChannel(Tensor pcm, int channel, int keepSamples)
    {
        int totalSamples = (int)pcm.Shape[2];
        float[] samples = new float[keepSamples];
        float* src = (float*)pcm.DataPointer + (long)channel * totalSamples;
        new ReadOnlySpan<float>(src, keepSamples).CopyTo(samples);
        return samples;
    }
}
