using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Models;
using SharpInference.Core.Pipelines;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Music;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>ACE-Step v1.5 turbo text/lyrics-to-music pipeline — the 2B Qwen3-block flow-matching DiT over Oobleck
/// latents (25 Hz, 64-ch): pre-computed Qwen3-Embedding-0.6B prompt + lyric states → packed
/// [lyric ‖ timbre ‖ text] cross-attention conditions → fixed 8-step shift-3 Euler ODE (hardcoded
/// <c>SHIFT_TIMESTEPS</c> table, <b>no CFG</b>, r = t) → Oobleck decode to 48 kHz stereo. Plain T2M uses the silence
/// latent as <c>src_latents</c> (VAE-encoded at first use when the injected decoder can encode — the reference ships
/// it as <c>silence_latent.pt</c>; recompute is the research doc's sanctioned alternative) with an all-ones chunk
/// mask. The FSQ/LM hint path (cover mode, Comfy's <c>generate_audio_codes</c>) is phase 2 — <c>lmHints</c> is its
/// null-default extension point. <b>Status: built, first-run validation pending</b> — numerics unverified vs the
/// reference; hints-less quality vs Comfy defaults is an open question (research §7).</summary>
public sealed unsafe class AceStepPipeline15 : DiffusionPipelineBase
{
    private readonly AceStep15Dit _dit;
    private readonly AceStep15ConditionEncoder _encoder;
    private readonly IAudioLatentDecoder _vae;
    private readonly AceStep15Config _config;
    private Tensor? _silenceFrames;

    public AceStepPipeline15(IBackend backend, AceStep15Dit dit, AceStep15ConditionEncoder encoder,
        IAudioLatentDecoder vae, AceStep15Config config)
        : base(backend)
    {
        _dit = dit ?? throw new ArgumentNullException(nameof(dit));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>Generates stereo audio from pre-computed Qwen3-Embedding-0.6B prompt states <c>[T_text, 1024]</c>
    /// and optional lyric states <c>[L, 1024]</c> (null → no lyric tokens; instrumental via prompt tags).
    /// <paramref name="timbreLatent"/> is an optional reference-audio Oobleck latent <c>[T_ref, 64]</c>;
    /// <paramref name="lmHints"/> is the phase-2 cover-mode hook (25 Hz detokenizer latents <c>[1, T, 64]</c>
    /// replacing the silence src). Returns one <c>float[]</c> per channel at 48 kHz.</summary>
    public (float[] Left, float[] Right, int SampleRate, int Seed) Generate(
        Tensor textHidden, Tensor? lyricHidden, double durationSeconds,
        float? shift = null, int? seed = null, Tensor? timbreLatent = null, Tensor? lmHints = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (durationSeconds < 1 || durationSeconds > 600)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "duration must be 1..600 s.");

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int patch = _config.PatchSize;
        int frames = Math.Max(4 * patch, (_config.LatentFrames(durationSeconds) + patch - 1) / patch * patch);
        float[] timesteps = AceStep15Config.GetTimesteps(shift ?? _config.FlowShift);
        int steps = timesteps.Length;

        Logs.Info($"ACE-Step 1.5 turbo: {durationSeconds:0}s ({frames} latent frames), {steps} steps, " +
            $"shift={shift ?? _config.FlowShift}, lyrics={(lyricHidden is null ? 0 : lyricHidden.Shape[0])} tokens, " +
            $"timbre={(timbreLatent is not null)}, hints={(lmHints is not null)}, seed={actualSeed}");
        Logs.Warning("ACE-Step 1.5 pipeline is first-run-validation pending — numerics unverified vs the reference.");

        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor conditions = _encoder.EncodeConditions(Backend, textHidden, lyricHidden, timbreLatent);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());

        Tensor context = BuildContextLatents(frames, lmHints);
        Tensor z = SeedGenerator.CreateNoise(new TensorShape(1, frames, _config.LatentChannels), actualSeed);

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            for (int i = 0; i < steps; i++)
            {
                float sigma = timesteps[i];
                float sigmaNext = i < steps - 1 ? timesteps[i + 1] : 0f;   // final step integrates to 0
                Tensor v = _dit.Forward(Backend, z, context, conditions, sigma, sigma);   // turbo: r = t
                float* zp = (float*)z.DataPointer;
                float* vp = (float*)v.DataPointer;
                float dt = sigmaNext - sigma;
                for (long e = 0; e < z.Shape.ElementCount; e++) zp[e] += dt * vp[e];
                v.Dispose();
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, sw.Elapsed.TotalMilliseconds));
            }
        }
        finally
        {
            conditions.Dispose();
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

        Logs.Info($"ACE-Step 1.5 complete: {channelSamples / (double)_config.SampleRate:0.0}s stereo, seed={actualSeed}");
        return (left, right, _config.SampleRate, actualSeed);
    }

    /// <summary>Builds <c>context_latents [1, T, 128]</c> = src ‖ chunk-mask per frame: src = phase-2
    /// <paramref name="lmHints"/> when given, else the silence latent tiled; mask = all ones (generate everything —
    /// repaint regions are a later feature).</summary>
    private Tensor BuildContextLatents(int frames, Tensor? lmHints)
    {
        int latCh = _config.LatentChannels;
        if (lmHints is not null && (lmHints.Shape.Rank != 3 || lmHints.Shape[1] != frames || (int)lmHints.Shape[2] != latCh))
            throw new ArgumentException($"lm hints must be [1, {frames}, {latCh}]; got {lmHints.Shape}.", nameof(lmHints));
        if (lmHints is null) EnsureSilenceFrames();

        Tensor context = new Tensor(new TensorShape(1, frames, 2 * latCh), DType.F32);
        float* cp = (float*)context.DataPointer;
        float* hp = lmHints is not null ? (float*)lmHints.DataPointer : (float*)_silenceFrames!.DataPointer;
        int srcPeriod = lmHints is not null ? frames : (int)_silenceFrames!.Shape[0];
        for (int i = 0; i < frames; i++)
        {
            long rowOff = (long)i * 2 * latCh;
            Buffer.MemoryCopy(hp + (long)(i % srcPeriod) * latCh, cp + rowOff, latCh * 4, latCh * 4);
            for (int c = 0; c < latCh; c++) cp[rowOff + latCh + c] = 1f;
        }
        return context;
    }

    /// <summary>Computes the silence src latent once: 1 s of digital silence through the VAE encoder (the research
    /// doc's sanctioned recompute of the shipped <c>silence_latent.pt</c>). Falls back to zero latents when the
    /// injected decoder cannot encode — validation-gated: parity with the .pt asset is unverified either way until a
    /// real-checkpoint dump comparison.</summary>
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
    }
}
