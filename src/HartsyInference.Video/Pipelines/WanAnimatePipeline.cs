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

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan-Animate (character animation) pipeline. Drives the <see cref="WanAnimateTransformer"/>: a pose video is
/// VAE-encoded into the pose-conditioning latent (added to the non-first latent frames in-DiT), and a face video
/// (pixel space, fixed motion-encoder resolution) drives the motion-encoder → face-encoder → face-adapter pathway.
/// Flow-match Euler + 2-way text CFG; reuses the Wan VAE (z=16).
///
/// <para><b>Status:</b> structural — the reference-image latent concat and replace-mode background/mask conditioning
/// are NOT modeled here (pose + face conditioning only); numerics validation-pending vs the real checkpoint.</para></summary>
public sealed unsafe class WanAnimatePipeline : DiffusionPipelineBase
{
    private readonly WanAnimateTransformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

    public WanAnimatePipeline(IBackend backend, WanAnimateTransformer transformer, IWanVaeDecoder vae,
        IWanVaeEncoder encoder, WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>Generates an animated clip. <paramref name="poseRgbClip"/> is <c>[1, 3, T, H, W]</c> (the driving
    /// pose/skeleton video); <paramref name="faceRgbClip"/> is <c>[1, 3, Tface, size, size]</c> at the motion-encoder
    /// resolution (the driving face video). Output length follows the pose clip's latent grid.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateAnimation(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor poseRgbClip, Tensor faceRgbClip,
        TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        int pixT = (int)poseRgbClip.Shape[2], pixH = (int)poseRgbClip.Shape[3], pixW = (int)poseRgbClip.Shape[4];
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"pose H/W must be divisible by {sp}.");
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"pose frame count must satisfy (T-1) % {tp} == 0; got {pixT}.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int tLat = (pixT - 1) / tp + 1, hLat = pixH / sp, wLat = pixW / sp, latentCh = _config.VaeLatentChannels;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        float shift = _config.FlowShift;

        Logs.Info($"Wan-Animate: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}, seed={seed} " +
            $"(latent {latentCh}x{tLat}x{hLat}x{wLat})");
        Logs.Warning("Wan-Animate pipeline is first-run-validation pending — reference/background conditioning not modeled.");

        // Pose conditioning: VAE-encode the pose clip, then keep the non-first latent frames (the DiT adds them to
        // latent frames 1..gt-1).
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        Tensor poseLatentFull = _encoder.Encode(Backend, poseRgbClip);   // [1, z, tLat, hLat, wLat]
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        Tensor pose = DropFirstLatentFrame(poseLatentFull, latentCh, tLat, hLat, wLat);   // [1, z, tLat-1, hLat, wLat]
        poseLatentFull.Dispose();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Tensor latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tLat, hLat, wLat]), seed);
        // VALIDATION-PENDING: Wan 2.2 UniPC scheduler (solver_order=2, bh2, predict_x0, flow sigmas, exponential
        // shift) — verify the Animate pose/face path's UniPC trajectory vs the diffusers Wan Animate pipeline.
        FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
        scheduler.SetTimesteps(steps, shift);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float tEmb = scheduler.Timesteps[k];
            Tensor vCond = _transformer.Forward(Backend, latents, pose, faceRgbClip, promptEmbeds, tEmb);
            Tensor vUncond = _transformer.Forward(Backend, latents, pose, faceRgbClip, negativeEmbeds, tEmb);
            LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
            scheduler.Step(latents, vCond);
            vCond.Dispose(); vUncond.Dispose();
            sw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
            {
                Latent = latents,
                LatentArch = LatentArchitecture.Wan,
            });
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        pose.Dispose();

        Tensor rgb;
        try { rgb = _vae.Decode(Backend, latents); }
        finally { latents.Dispose(); }
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Animate complete ({frames.Length} frames, seed={seed})");
        return (frames, pixW, pixH, seed);
    }

    /// <summary>Drops latent frame 0 → <c>[1, C, T-1, H, W]</c> (the pose conditioning that the DiT adds to frames 1..).</summary>
    private static Tensor DropFirstLatentFrame(Tensor x, int c, int t, int h, int w)
    {
        Tensor o = new Tensor(new TensorShape([1L, c, t - 1, h, w]), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(xp + ((long)ci * t + 1) * frame, op + (long)ci * (t - 1) * frame, (long)(t - 1) * frame * 4, (long)(t - 1) * frame * 4);
        return o;
    }
}
