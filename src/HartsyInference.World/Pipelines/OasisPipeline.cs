using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.World.ActionEncoders;
using HartsyInference.Video;

namespace HartsyInference.World.Pipelines;

/// <summary>Oasis-500m (Decart/Etched, MIT) action-conditioned world model — autoregressive in time, diffusion in
/// space, matching <c>open-oasis/generate.py</c>: VAE-encode the prompt frame, then per new frame append clamped
/// Gaussian noise and run 10 v-pred DDIM steps where committed context frames are held at the Diffusion-Forcing
/// stabilization noise level (14) while only the target frame walks the schedule and is written back. A sliding
/// window caps the visible context at 32 latent frames. Actions are 25-dim VPT rows per frame (a zero row is
/// prepended internally so action <c>t</c> is "the input that produced frame <c>t</c>").
///
/// <para>This is the Phase 10 pedagogical reference model — full RGB-in/RGB-out in pure C# (the ViT-VAE encoder ships
/// with the checkpoint). <b>Numerics validation-pending</b>; the 360×640 resolution is the only one the checkpoint
/// supports. See <c>docs/Research/OASIS_ARCHITECTURE.md</c>.</para></summary>
public sealed unsafe class OasisPipeline : DiffusionPipelineBase
{
    /// <summary>Latent scaling factor (20/255) from <c>generate.py</c>.</summary>
    public const float ScalingFactor = 0.07843137255f;

    private readonly OasisDit _dit;
    private readonly OasisVitVae _vae;
    private readonly DdimVPredScheduler _scheduler = new();

    public OasisPipeline(IBackend backend, OasisDit dit, OasisVitVae vae)
        : base(backend)
    {
        _dit = dit;
        _vae = vae;
    }

    /// <summary>Generates <paramref name="totalFrames"/> frames (including the prompt frame) from an RGB24 prompt
    /// image and a per-frame action plan (<c>[totalFrames − 1][25]</c> rows — the prompt frame gets the prepended
    /// zero action). Returns interleaved-RGB frames.</summary>
    public (byte[][] frames, int width, int height, int seed) Generate(
        ReadOnlySpan<byte> promptRgb24, int width, int height, float[][] actions, int totalFrames,
        int ddimSteps = 10, int stabilizationLevel = 15, float noiseAbsMax = 20f, int? seed = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int p = _vae.PatchSize;
        if (width % p != 0 || height % p != 0)
            throw new ArgumentException($"Width/height must be divisible by the VAE patch size {p}.");
        if (totalFrames < 2) throw new ArgumentOutOfRangeException(nameof(totalFrames));
        if (actions.Length < totalFrames - 1)
            throw new ArgumentException($"Need ≥{totalFrames - 1} action rows; got {actions.Length}.", nameof(actions));
        if (promptRgb24.Length < width * height * 3)
            throw new ArgumentException($"prompt must be {width * height * 3} RGB bytes.", nameof(promptRgb24));

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        int gh = height / p, gw = width / p;
        int maxFrames = _dit.Config.MaxFrames;
        int c = _dit.Config.InChannels;

        Logs.Info($"Oasis: {totalFrames} frames {width}x{height}, {ddimSteps} DDIM steps/frame, stabilization {stabilizationLevel}, seed {actualSeed}");
        Logs.Warning("Oasis pipeline is first-run-validation pending — numerics unverified vs the reference.");

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Backend.PreloadWeights(_vae.EnumerateWeights());

        // Prompt frame → scaled latent.
        Tensor promptRgb = RgbToTensor(promptRgb24, width, height);
        Tensor promptLatent = _vae.Encode(Backend, promptRgb);
        promptRgb.Dispose();
        Scale(promptLatent, ScalingFactor);

        // Per-frame latents [c, gh, gw]; actions with the prepended zero row.
        List<Tensor> latents = [ToFrame(promptLatent, c, gh, gw)];
        promptLatent.Dispose();
        float[][] alignedActions = new float[totalFrames][];
        alignedActions[0] = new float[OasisActionEncoder.ActionDim];
        for (int i = 1; i < totalFrames; i++) alignedActions[i] = actions[i - 1];

        int[] noiseRange = _scheduler.BuildNoiseRange(ddimSteps);
        try
        {
            for (int i = 1; i < totalFrames; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor noise = SeedGenerator.CreateNoise(new TensorShape([(long)c, gh, gw]), actualSeed + i);
                Clamp(noise, noiseAbsMax);
                latents.Add(noise);

                int startFrame = Math.Max(0, i + 1 - maxFrames);
                int window = i + 1 - startFrame;
                int[] noiseIdx = new int[window];
                for (int f = 0; f < window - 1; f++) noiseIdx[f] = stabilizationLevel - 1;

                for (int step = ddimSteps; step >= 1; step--)
                {
                    int t = noiseRange[step];
                    int tNext = noiseRange[step - 1];
                    noiseIdx[window - 1] = t;

                    Tensor windowLatents = StackFrames(latents, startFrame, window, c, gh, gw);
                    Tensor windowActions = StackActions(alignedActions, startFrame, window);
                    Tensor v = _dit.Forward(Backend, windowLatents, noiseIdx, windowActions);
                    windowLatents.Dispose();
                    windowActions.Dispose();

                    Tensor vTarget = SliceFrame(v, window - 1, c, gh, gw);
                    v.Dispose();
                    _scheduler.StepFrame(latents[i], vTarget, t, tNext);
                    vTarget.Dispose();
                }
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(i, totalFrames - 1, sw.Elapsed.TotalMilliseconds));
            }

            byte[][] frames = new byte[totalFrames][];
            for (int i = 0; i < totalFrames; i++)
            {
                Tensor unscaled = ToLatent4d(latents[i], c, gh, gw);
                Scale(unscaled, 1f / ScalingFactor);
                Tensor rgb = _vae.Decode(Backend, unscaled);
                unscaled.Dispose();
                frames[i] = RgbToBytes(rgb, width, height);
                rgb.Dispose();
            }
            Logs.Info($"Oasis rollout complete ({totalFrames} frames, seed={actualSeed})");
            return (frames, width, height, actualSeed);
        }
        finally
        {
            foreach (Tensor frame in latents) frame.Dispose();
            Backend.Sync();
            Backend.FreeWeights(_dit.EnumerateWeights());
            Backend.FreeWeights(_vae.EnumerateWeights());
        }
    }

    private Tensor RgbToTensor(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        Tensor rgb = new Tensor(new TensorShape([1L, 3, height, width]), DType.F32);
        float* p = (float*)rgb.DataPointer;
        long frame = (long)height * width;
        for (long pix = 0; pix < frame; pix++)
            for (int ci = 0; ci < 3; ci++)
                p[ci * frame + pix] = rgb24[(int)(pix * 3 + ci)] / 127.5f - 1f;
        return rgb;
    }

    private static byte[] RgbToBytes(Tensor rgb, int width, int height)
    {
        byte[] outB = new byte[width * height * 3];
        float* p = (float*)rgb.DataPointer;
        long frame = (long)height * width;
        for (long pix = 0; pix < frame; pix++)
            for (int ci = 0; ci < 3; ci++)
            {
                int b = (int)MathF.Round((p[ci * frame + pix] + 1f) * 127.5f);
                outB[pix * 3 + ci] = (byte)Math.Clamp(b, 0, 255);
            }
        return outB;
    }

    private static Tensor ToFrame(Tensor latent4d, int c, int gh, int gw)
    {
        Tensor o = new Tensor(new TensorShape([(long)c, gh, gw]), DType.F32);
        long n = (long)c * gh * gw;
        Buffer.MemoryCopy((float*)latent4d.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
        return o;
    }

    private static Tensor ToLatent4d(Tensor frame, int c, int gh, int gw)
    {
        Tensor o = new Tensor(new TensorShape([1L, c, gh, gw]), DType.F32);
        long n = (long)c * gh * gw;
        Buffer.MemoryCopy((float*)frame.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
        return o;
    }

    private static Tensor StackFrames(List<Tensor> latents, int start, int count, int c, int gh, int gw)
    {
        Tensor o = new Tensor(new TensorShape([(long)count, c, gh, gw]), DType.F32);
        long per = (long)c * gh * gw;
        float* op = (float*)o.DataPointer;
        for (int f = 0; f < count; f++)
            Buffer.MemoryCopy((float*)latents[start + f].DataPointer, op + f * per, per * 4, per * 4);
        return o;
    }

    private static Tensor StackActions(float[][] actions, int start, int count)
    {
        Tensor o = new Tensor(new TensorShape(count, OasisActionEncoder.ActionDim), DType.F32);
        float* op = (float*)o.DataPointer;
        for (int f = 0; f < count; f++)
            for (int d = 0; d < OasisActionEncoder.ActionDim; d++)
                op[(long)f * OasisActionEncoder.ActionDim + d] = actions[start + f][d];
        return o;
    }

    private static Tensor SliceFrame(Tensor stacked, int frame, int c, int gh, int gw)
    {
        Tensor o = new Tensor(new TensorShape([(long)c, gh, gw]), DType.F32);
        long per = (long)c * gh * gw;
        Buffer.MemoryCopy((float*)stacked.DataPointer + (long)frame * per, (float*)o.DataPointer, per * 4, per * 4);
        return o;
    }

    private static void Scale(Tensor x, float factor)
    {
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= factor;
    }

    private static void Clamp(Tensor x, float absMax)
    {
        float* p = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = Math.Clamp(p[i], -absMax, absMax);
    }
}
