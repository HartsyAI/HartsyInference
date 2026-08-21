using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Lance (ByteDance, Apache-2.0) text-to-video pipeline. Reuses the entire image stack — <see cref="LanceTransformer"/> (the forward pass is T-agnostic), <see cref="LancePipelineCommon"/> sequence builders, <see cref="Wan22VaeDecoder"/> (streaming T&gt;1 decode) — adding only the time axis.
///
/// <para>Deltas vs <c>LanceImagePipeline</c>: <c>timestep_shift = 4.0</c>; token grid is <c>(T_lat, H/16, W/16)</c> with the vision block's temporal M-RoPE axis stepping by <see cref="LanceConfig.TokensPerSecond"/> per latent frame; and the VAE decode streams across latent frames. <c>num_frames</c> must satisfy <c>(num_frames − 1) % 4 == 0</c> (4× temporal compression, max 121), giving <c>T_lat = (num_frames − 1)/4 + 1</c>.</para>
///
/// <para><b>Status: awaits the Lance_3B_Video checkpoint</b> — the image variant's backbone is real-weight reconciled; the video sampling defaults follow the same upstream inference script (2-way text CFG in <c>(0.4,1]</c> with global renorm).</para></summary>
public sealed unsafe class LanceVideoPipeline : DiffusionPipelineBase
{
    private const int VaeChannels = 48;

    private readonly LanceTransformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly LanceConfig _config;
    private readonly LancePromptTemplate _template;

    public LanceVideoPipeline(IBackend backend, LanceTransformer transformer, Wan22VaeDecoder vae, LanceConfig config,
        LancePromptTemplate? promptTemplate = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _template = promptTemplate ?? LancePromptTemplate.Bare(config);
    }

    /// <summary>Generates <paramref name="numFrames"/> RGB frames from raw caption token ids, returning all frames at once. For memory-bounded streaming use <see cref="GenerateFramesAsync"/>.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null)
    {
        Tensor vaeLatent = RunDenoise(promptTokenIds, negativeTokenIds, request, numFrames, onProgress, out int seed);
        Tensor rgb;
        try { rgb = _vae.Decode(Backend, vaeLatent); }
        finally { vaeLatent.Dispose(); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = FrameToBytes(rgb, i);
        rgb.Dispose();
        Logs.Info($"Lance T2V complete ({frames.Length} frames, seed={seed})");
        return (frames, request.Width ?? 848, request.Height ?? 480, seed);
    }

    /// <summary>Streams decoded frames one at a time (pull-based → memory bounded to a single VAE frame-group; natural backpressure). Pair with an <c>IVideoEncoder</c> to write an MP4/frame-sequence without holding the whole clip.</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(
        int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Tensor vaeLatent = RunDenoise(promptTokenIds, negativeTokenIds, request, numFrames, onProgress, out _);
        try
        {
            int idx = 0;
            foreach (Tensor group in _vae.DecodeStreaming(Backend, vaeLatent))   // [1,3,groupT,H,W] per latent frame
            {
                int gT = (int)group.Shape[2], h = (int)group.Shape[3], w = (int)group.Shape[4];
                for (int gi = 0; gi < gT; gi++)
                    yield return new VideoFrame(idx++, w, h, FrameToBytes(group, gi));
                group.Dispose();
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
            }
        }
        finally
        {
            vaeLatent.Dispose();
        }
    }

    /// <summary>Runs the shared denoise loop and returns the VAE-ready latent <c>[1,48,gridT,vaeLatentH,vaeLatentW]</c>. The transformer weights are freed before return.</summary>
    private Tensor RunDenoise(int[] promptTokenIds, int[] negativeTokenIds, TextToImageRequest request, int numFrames,
        Action<GenerationProgress>? onProgress, out int seed)
    {
        ThrowIfDisposed();
        seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? 848, height = request.Height ?? 480;
        int downscale = _config.VaeDownsampleSpatial * _config.LatentPatchSize.H;
        if (width % downscale != 0 || height % downscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {downscale} for Lance video.");
        if (numFrames < 1 || (numFrames - 1) % _config.VaeDownsampleTemporal != 0)
            throw new ArgumentException($"num_frames must satisfy (num_frames-1) % {_config.VaeDownsampleTemporal} == 0 (e.g. 1, 5, 9, … 121); got {numFrames}.");

        int gridT = (numFrames - 1) / _config.VaeDownsampleTemporal + 1;
        int gridH = height / downscale, gridW = width / downscale;
        int nVae = gridT * gridH * gridW;
        int steps = request.Steps ?? _config.NumTimesteps;
        float cfg = request.CfgScale ?? _config.CfgTextScale;
        float shift = _config.VideoTimestepShift;

        Logs.Info($"Lance T2V denoise: {numFrames}f {width}x{height}, {steps} steps, cfg={cfg}, seed={seed} (grid {gridT}x{gridH}x{gridW}, {nVae} tokens)");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        using LanceSequence cond = LancePipelineCommon.BuildGenSequence(_template, promptTokenIds, _config, gridT, gridH, gridW);
        using LanceSequence uncond = LancePipelineCommon.BuildGenSequence(_template, negativeTokenIds, _config, gridT, gridH, gridW);
        int[] latentPosIds = LancePipelineCommon.BuildLatentPositionIds(gridT, gridH, gridW, _config.MaxLatentSize);

        Tensor latents = SeedGenerator.CreateNoise(new TensorShape(nVae, _config.PatchFeatureDim), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, shift);
        // Sampler selection (2026-08-20). Lance's own shifted grid IS the sigma array the sampler integrates —
        // descending 1→0 with the terminal zero — so it goes to the resolver directly. Rebuilding it through a
        // lookalike FlowMatch scheduler would differ by an ulp for general (i, N) and shift every default
        // generation. Text-to-video only: this pipeline has no img2img/inpaint entry, so the init latent is always
        // pure noise at sigma[0] and `startsFromNoisedInit` stays false.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, tsteps, seed, "Lance");

        // Because the timesteps double as the sigmas, the value the transformer conditions on is the sampler's own
        // sigma — no ×1000 round trip to reproduce, and a second-order sub-step conditions (and gates its CFG
        // interval) at the sigma it is actually evaluating rather than at the enclosing step's.
        DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
            PredictionType.FlowVelocity,
            (x, s, _) =>
            {
                Tensor vCond = _transformer.Forward(Backend, cond.TextTokenIds, x, latentPosIds, s,
                    cond.PositionIds, cond.UndIdx, cond.GenIdx, cond.AttentionMask);
                if (cfg > 1f && s > _config.CfgIntervalMin)
                {
                    Tensor vUncond = _transformer.Forward(Backend, uncond.TextTokenIds, x, latentPosIds, s,
                        uncond.PositionIds, uncond.UndIdx, uncond.GenIdx, uncond.AttentionMask);
                    // The global renorm is a NON-linear combine, so it cannot ride the pair's guidance scale.
                    // Fold it here and hand the sampler an already-combined velocity as both halves at guidance 1,
                    // which makes the fused op's combine an exact identity.
                    LancePipelineCommon.CfgRenormGlobalInPlace(vCond, vUncond, cfg, _config.CfgRenormMin);
                    vUncond.Dispose();
                }
                return new DenoisePrediction(vCond, vCond);
            });
        sampler.Reset(latents.Shape);

        for (int k = 0; k < steps; k++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            sampler.Step(Backend, latents, predictor, k);
            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(k + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        (int pt, int ph, int pw) = _config.LatentPatchSize;
        Tensor latentCl = LanceLatentPatch.Unpatchify(latents, gridT, gridH, gridW, pt, ph, pw, VaeChannels);
        latents.Dispose();
        Tensor vaeLatent = LancePipelineCommon.ChannelLastToBcthw(latentCl);   // [1,48,gridT,vaeLatentH,vaeLatentW]
        latentCl.Dispose();
        return vaeLatent;
    }

    private static byte[] FrameToBytes(Tensor rgb, int frameIndex) => VideoRgbFrames.ExtractFrame(rgb, frameIndex);
}
