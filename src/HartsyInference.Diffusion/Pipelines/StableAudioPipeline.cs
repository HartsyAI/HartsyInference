using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Stable Audio Open Small text-to-audio/SFX pipeline: pre-computed T5-base prompt states + a <c>seconds_total</c> timing token → <see cref="StableAudioDit"/> rectified-flow denoise (8-step <see cref="StableAudioPingPongScheduler"/>, no CFG — ARC-distilled) → Oobleck VAE decode → 44.1 kHz stereo, trimmed to the requested duration. The DiT always denoises the full trained-length latent (<see cref="StableAudioDitConfig.MaxLatentTokens"/>, ~11.89 s) regardless of the requested <c>secondsTotal</c> — duration is controlled purely through the timing conditioning token and a post-decode trim, matching upstream <c>generate_diffusion_cond</c>. <b>Status: DiT / VAE / timing conditioner components individually real-weight-verified (cosine 1.0 each, see PARITY_VERIFICATION.md); this composition is structurally wired but not yet end-to-end validated against a Python reference.</b></summary>
public sealed unsafe class StableAudioPipeline : DiffusionPipelineBase
{
    private readonly StableAudioDit _dit;
    private readonly IAudioLatentDecoder _vae;
    private readonly StableAudioNumberEmbedder _timingConditioner;
    private readonly StableAudioDitConfig _config;

    public StableAudioPipeline(IBackend backend, StableAudioDit dit, IAudioLatentDecoder vae,
        StableAudioNumberEmbedder timingConditioner, StableAudioDitConfig config)
        : base(backend)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _timingConditioner = timingConditioner ?? throw new ArgumentNullException(nameof(timingConditioner));
        _config = config ?? throw new ArgumentNullException(nameof(config));
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

    private static float[] TrimChannel(Tensor pcm, int channel, int keepSamples)
    {
        int totalSamples = (int)pcm.Shape[2];
        float[] samples = new float[keepSamples];
        float* src = (float*)pcm.DataPointer + (long)channel * totalSamples;
        new ReadOnlySpan<float>(src, keepSamples).CopyTo(samples);
        return samples;
    }
}
