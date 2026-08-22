using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>Wan-Animate-2 (character animation driven by a raw video) pipeline. One chunk per call: the reference
/// image becomes latent frame 0 of the generation stream, the driving video's latents run through the DiT once at a
/// fixed timestep to fill the per-block driving cache, and the denoise loop splices its K/V in frame-aligned.
/// There is no pose render, no face crop, no motion encoder and no retargeting — every V1 conditioning surface is
/// gone, and the driving pixels go straight to the VAE.
///
/// <para><b>CFG runs sequentially, deliberately.</b> <see cref="WanAnimate2Transformer"/>'s RoPE/text caches and its
/// log-scale bias are unsynchronized, so the two branches must not be dispatched to concurrent backends; this
/// pipeline never accepts a <see cref="DiffusionPipelineBase.CfgParallelBackend"/> and its recipe warns when one is
/// configured.</para>
///
/// <para><b>Sampling settings are build-specific and NOT interchangeable.</b> The base checkpoint wants 40 steps
/// at guidance 3.0 (upstream's demo and yaml); the distillation checkpoint wants ~6-10 steps at guidance 1.0.
/// Running the base build at the distillation build's settings renders hazy and progressively loses the clip as
/// the frame count grows — which is not a defect in the driving splice, however much it looks like one.</para></summary>
public sealed unsafe class WanAnimate2Pipeline : DiffusionPipelineBase
{
    /// <summary>The reference's clip length: 81 pixel frames, of which the last-but-one chunk boundary carries
    /// exactly one frame forward.</summary>
    public const int ClipLength = 81;

    /// <summary>Pixel frames of overlap between consecutive chunks (upstream's <c>first_num</c>).</summary>
    public const int ChunkOverlapFrames = 1;

    /// <summary>Upstream's <c>sample_shift</c>, applied to the sigmas.</summary>
    public const float DefaultFlowShift = 5f;

    /// <summary>The reference solver, and this pipeline's default — <see cref="FlowDpmPlusPlus2MScheduler"/> over
    /// <c>get_sampling_sigmas</c>.</summary>
    public const string DefaultSampler = "dpm++2m";

    /// <summary>The one alternative the pipeline will run. Selectable, never the default: UniPC's sigma grid differs
    /// from <c>get_sampling_sigmas</c> as well as its solver, so it is not a parity reference.</summary>
    public const string AlternateSampler = "unipc";

    private readonly WanAnimate2Transformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

    /// <summary>Opt-in halving of the driving cache to BF16. Off because the rest of the forward is F32, so it is a
    /// divergence from the reference and from ComfyUI alike, and it has had no real-weight quality run.</summary>
    public const string Bf16DrivingCacheSwitch = "HARTSY_ANIMATE2_BF16_DRIVING_CACHE";

    /// <summary>Measured per-token activation slope of the Animate denoise loop, reused here — the block internals
    /// are the same Wan i2v ones, and the binding constraint is activations rather than weights.</summary>
    private const long ActivationBytesPerToken = 671_089;

    /// <summary>Allowance for cuBLAS workspace, the prefetch window and pool slack, on top of the token-scaled term.</summary>
    private const long FixedHeadroomBytes = 1L << 30;

    public WanAnimate2Pipeline(IBackend backend, WanAnimate2Transformer transformer, IWanVaeDecoder vae,
        IWanVaeEncoder encoder, WanVideoConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _encoder = encoder;
        _config = config;
    }

    /// <summary>Upstream's base-build sampling (`wan_animate_2_demo.py`): 40 steps at guidance 3.0.</summary>
    public const int BaseBuildSteps = 40;

    /// <inheritdoc cref="BaseBuildSteps"/>
    public const float BaseBuildGuidance = 3.0f;

    /// <summary>Upstream's distillation-build sampling (README): 10 steps at guidance 1.0.</summary>
    public const int DistillBuildSteps = 10;

    /// <summary>Fewest steps the base build is trusted at. Below this it renders progressively hazier as the clip
    /// lengthens, which reads as the driving video being ignored rather than as under-denoising.</summary>
    private const int BaseBuildMinSteps = 20;

    /// <summary>The two builds want incompatible sampling settings and nothing in a checkpoint announces which one
    /// it is except its file name, so a caller can silently run one at the other's settings and get a plausible
    /// short clip and mush at full length. Returns the warning to log, or null when the settings suit the build.
    /// <paramref name="distillBuild"/> is <see cref="WanVideoConfig.Animate2LogScale"/> being non-zero, which is
    /// exactly what <c>WanAnimate2Transformer.ResolveLogScale</c> routed off the file name.</summary>
    internal static string? SettingsMismatchWarning(bool distillBuild, int steps, float guidance)
    {
        if (distillBuild)
        {
            return guidance <= 1f ? null
                : $"[WanAnimate2] This is the DISTILLATION build; it samples at guidance 1.0 and ~{DistillBuildSteps} "
                    + $"steps. Guidance {guidance} will over-drive it.";
        }
        if (steps >= BaseBuildMinSteps && guidance > 1f) return null;
        return $"[WanAnimate2] This is the BASE build, which upstream samples at {BaseBuildSteps} steps and guidance "
            + $"{BaseBuildGuidance}; this request is {steps} steps at guidance {guidance}. Those are the DISTILLATION "
            + "build's settings and the base weights render hazy at them — the haze grows with the frame count, so a "
            + "short clip can look fine while a full-length one does not. Raise the steps and the guidance, or load "
            + "the *_distill_* checkpoint.";
    }

    /// <summary>True when the denoise must run the negative branch at all. At or below 1.0 the CFG fold is the
    /// identity, and the reference takes a single forward — which also means the block-9 uncond skip never fires
    /// for the distillation build.</summary>
    public static bool UsesCfgBranch(float guidance) => guidance > 1f;

    /// <summary>Normalizes a requested sampler name to one this pipeline runs, throwing on anything else. Null or
    /// blank selects <see cref="DefaultSampler"/>; upstream offers no other solver, so the only alternative is the
    /// engine's own UniPC.</summary>
    public static string ResolveSampler(string? requested)
    {
        if (string.IsNullOrWhiteSpace(requested)) return DefaultSampler;
        if (string.Equals(requested, DefaultSampler, StringComparison.OrdinalIgnoreCase)) return DefaultSampler;
        if (string.Equals(requested, AlternateSampler, StringComparison.OrdinalIgnoreCase)) return AlternateSampler;
        throw new NotSupportedException(
            $"Wan-Animate-2 has no '{requested}' implementation. Upstream samples with FlowDPM++ 2M midpoint over "
            + $"get_sampling_sigmas ('{DefaultSampler}', the default); '{AlternateSampler}' is the only alternative.");
    }

    /// <summary>Latent frames the generation stream carries for a <paramref name="pixelFrames"/>-frame chunk: the
    /// causal VAE's <c>(T-1)/4 + 1</c> plus the prepended reference slot. The driving stream gets one fewer, which is
    /// the invariant <see cref="WanAnimate2Transformer.EncodeDriving"/> enforces.</summary>
    public static int GenerationLatentFrames(int pixelFrames, int temporalCompression) =>
        (pixelFrames - 1) / temporalCompression + 1 + 1;

    /// <summary>Renders one chunk. <paramref name="referenceRgb"/> is the character image <c>[1,3,1,H,W]</c> in
    /// [-1,1]; <paramref name="drivingRgbClip"/> is the raw driving video <c>[1,3,T,H,W]</c> in [-1,1], already
    /// fps-resampled and resized — no pose or skeleton rendering. <paramref name="carriedRgbFrame"/> is the previous
    /// chunk's last decoded frame <c>[1,3,1,H,W]</c>, which both seeds the continuation encode and flips latent
    /// frame 1's mask to known; null on the first chunk. <paramref name="drivingClipEmbeds"/> must be the CLIP-ViT-H
    /// state of <b>this chunk's</b> driving frame 0, not the reference image's. <paramref name="sampler"/> is resolved
    /// by <see cref="ResolveSampler"/>; null takes the reference solver.</summary>
    public (byte[][] Frames, int Width, int Height, int Seed) GenerateChunk(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor drivingPromptEmbeds,
        Tensor referenceRgb, Tensor drivingRgbClip, TextToImageRequest request,
        Tensor? referenceClipEmbeds = null, Tensor? drivingClipEmbeds = null,
        Tensor? carriedRgbFrame = null, string? sampler = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(referenceRgb);
        ArgumentNullException.ThrowIfNull(drivingRgbClip);
        int latentCh = _config.VaeLatentChannels;
        int sp = _config.VaeSpatialCompression, tp = _config.VaeTemporalCompression;
        if (_config.InChannels != 2 * latentCh + WanAnimate2Conditioning.MaskChannels)
            throw new InvalidOperationException(
                $"Wan-Animate-2 expects InChannels == 2·z + 4 ({2 * latentCh + WanAnimate2Conditioning.MaskChannels}); got {_config.InChannels}.");

        if (drivingRgbClip.Shape.Rank != 5 || drivingRgbClip.Shape[1] != 3)
            throw new ArgumentException($"drivingRgbClip must be [1,3,T,H,W]; got {drivingRgbClip.Shape}.", nameof(drivingRgbClip));
        if (referenceRgb.Shape.Rank != 5 || referenceRgb.Shape[1] != 3 || referenceRgb.Shape[2] != 1)
            throw new ArgumentException($"referenceRgb must be [1,3,1,H,W]; got {referenceRgb.Shape}.", nameof(referenceRgb));
        int pixT = (int)drivingRgbClip.Shape[2], pixH = (int)drivingRgbClip.Shape[3], pixW = (int)drivingRgbClip.Shape[4];
        if (referenceRgb.Shape[3] != pixH || referenceRgb.Shape[4] != pixW)
            throw new ArgumentException($"referenceRgb must be {pixW}x{pixH} to share the driving stream's token grid.", nameof(referenceRgb));
        if (pixH % sp != 0 || pixW % sp != 0)
            throw new ArgumentException($"driving H/W must be divisible by {sp}.", nameof(drivingRgbClip));
        if ((pixT - 1) % tp != 0)
            throw new ArgumentException($"driving frame count must satisfy (T-1) % {tp} == 0; got {pixT}.", nameof(drivingRgbClip));
        if (carriedRgbFrame is not null
            && (carriedRgbFrame.Shape.Rank != 5 || carriedRgbFrame.Shape[2] != 1
                || carriedRgbFrame.Shape[3] != pixH || carriedRgbFrame.Shape[4] != pixW))
            throw new ArgumentException($"carriedRgbFrame must be [1,3,1,{pixH},{pixW}]; got {carriedRgbFrame.Shape}.", nameof(carriedRgbFrame));

        int hLat = pixH / sp, wLat = pixW / sp;
        int videoLatentFrames = (pixT - 1) / tp + 1;
        int tTotal = videoLatentFrames + 1;
        bool continuation = carriedRgbFrame is not null;

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int steps = request.Steps ?? _config.NumInferenceSteps;
        float guidance = request.CfgScale ?? _config.GuidanceScale;
        bool useCfg = UsesCfgBranch(guidance);
        float shift = (request as VideoGenerationRequest)?.FlowShift ?? DefaultFlowShift;
        string samplerName = ResolveSampler(sampler);

        Logs.Info($"Wan-Animate-2: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}{(useCfg ? "" : " single forward")}, "
            + $"seed={seed}, sampler={samplerName} (gen latent {latentCh}x{tTotal}x{hLat}x{wLat}, driving {tTotal - 1} frame(s)"
            + $"{(continuation ? ", continuation chunk" : "")}, log_scale={_config.Animate2LogScale}).");
        string? settingsWarning = SettingsMismatchWarning(_config.Animate2LogScale != 0f, steps, guidance);
        if (settingsWarning is not null) Logs.Warning(settingsWarning);
        if (samplerName == AlternateSampler)
        {
            Logs.Warning("[WanAnimate2] UniPC was requested. Its sigma grid floors at 1/1000 where get_sampling_sigmas "
                + "reaches 0, so the trajectory differs from upstream in the grid as well as the solver — this output "
                + "is not a numerical parity reference.");
        }

        Tensor? conditioning = null, drivingLatent = null, latents = null;
        WanAnimate2DrivingCache? driving = null;
        BlockStreamingScope? stream = null;
        try
        {
            // One causal VAE encode per stream. The continuation frame goes over the HEAD of a mid-grey clip and is
            // encoded WITH it: the Wan VAE is temporally causal, so encoding the carried frame separately and
            // splicing latents is not the same tensor.
            Stopwatch phase = Stopwatch.StartNew();
            Backend.ResetOpProfile();
            Backend.PreloadWeights(_encoder.EnumerateWeights());
            Tensor referenceLatent = _encoder.Encode(Backend, referenceRgb);
            Backend.Sync();
            double tRefMs = phase.Elapsed.TotalMilliseconds;
            Tensor videoPixels = BuildContinuationPixels(carriedRgbFrame, pixT, pixH, pixW);
            Tensor videoLatent = _encoder.Encode(Backend, videoPixels);
            videoPixels.Dispose();
            Backend.Sync();
            double tGreyMs = phase.Elapsed.TotalMilliseconds - tRefMs;
            Tensor drivingLatentDev = _encoder.Encode(Backend, drivingRgbClip);
            Backend.Sync();
            double tDriveMs = phase.Elapsed.TotalMilliseconds - tRefMs - tGreyMs;
            Backend.FreeWeights(_encoder.EnumerateWeights());
            Backend.DumpOpProfile("animate2-vaeencode");
            Logs.Info($"Wan-Animate-2 VAE encode: reference+preload {tRefMs / 1000.0:F1}s, continuation clip "
                + $"{tGreyMs / 1000.0:F1}s, driving clip {tDriveMs / 1000.0:F1}s.");

            conditioning = WanAnimate2Conditioning.BuildGenerationChannels(referenceLatent, videoLatent, continuation);
            referenceLatent.Dispose();
            videoLatent.Dispose();
            // Host-materialized: the driving latent has to survive the per-step activation sweeps between the
            // prepass and the loop, and it is a few MB.
            drivingLatent = HostCopy(drivingLatentDev);
            drivingLatentDev.Dispose();
            Backend.TrimMemoryPool();

            (int pt, int ph, int pw) = _config.PatchSize;
            (int T, int H, int W) genGrid = (tTotal / pt, hLat / ph, wLat / pw);
            long tokenLoad = (long)genGrid.T * genGrid.H * genGrid.W;
            bool bf16Cache = EnvSwitch.IsEnabled(Bf16DrivingCacheSwitch, false);
            // The driving cache is built after this placement is chosen but outlives every step of it, so it is
            // headroom, not activation churn. Omitting it let the planner keep all 40 blocks resident and then OOM
            // on the cache with the card genuinely full.
            long drivingCacheBytes = _transformer.DrivingCacheBytes(genGrid, bf16Cache);
            long headroomBytes = Math.Max(EnvSwitch.GetLong("HARTSY_ANIMATE2_HEADROOM_MB", 3072) * 1024 * 1024,
                tokenLoad * ActivationBytesPerToken + FixedHeadroomBytes) + drivingCacheBytes;
            // Opened before the prepass, not just around the loop: EncodeDriving is a full 40-block forward and
            // needs the same streaming budget the denoise steps do.
            stream = BlockStreamingScope.Open(new BlockStreamingOptions
            {
                Backend = Backend,
                Denoiser = _transformer,
                ModelName = "Wan-Animate-2",
                HeadroomBytes = headroomBytes,
                TokenLoad = tokenLoad,
            });

            // The driving stream sits outside the guidance loop entirely — it never sees the negative prompt, and it
            // is built once per chunk rather than once per step.
            _transformer.PoseStrength = (float)EnvSwitch.GetLong("HARTSY_ANIMATE2_POSE_STRENGTH_X100", 100) / 100f;
            driving = _transformer.EncodeDriving(Backend, drivingLatent, drivingPromptEmbeds, drivingClipEmbeds, genGrid,
                bf16Cache: bf16Cache);
            long drivingTokens = (long)driving.Frames * driving.TokensPerFrame;
            Logs.Info($"Wan-Animate-2 driving cache: {driving.StoredBytes / (1024.0 * 1024.0 * 1024.0):F2} GiB "
                + $"({driving.StoredBytes / (double)drivingTokens / (1024.0 * 1024.0):F4} MiB per driving token, "
                + $"{drivingTokens} tokens, {driving.StorageDType.Name}).");
            stream.EndStep();
            Backend.FreeActivations();
            Logs.Info($"Wan-Animate-2 driving prepass: {phase.Elapsed.TotalMilliseconds / 1000.0 - tRefMs / 1000.0 - tGreyMs / 1000.0 - tDriveMs / 1000.0:F1}s (placement + EncodeDriving).");
            Backend.DumpOpProfile("animate2-prepass");
            phase.Restart();

            latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tTotal, hLat, wLat]), seed);
            bool alternate = samplerName == AlternateSampler;
            FlowUniPCMultistepScheduler? unipc = alternate ? new FlowUniPCMultistepScheduler(solverOrder: 2) : null;
            FlowDpmPlusPlus2MScheduler? dpm = alternate ? null : new FlowDpmPlusPlus2MScheduler();
            unipc?.SetTimesteps(steps, shift);
            dpm?.SetTimesteps(steps, shift);
            float[] timesteps = (alternate ? unipc!.Timesteps : dpm!.Timesteps).ToArray();

            for (int k = 0; k < steps; k++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                float tEmb = timesteps[k];
                Tensor modelInput = WanAnimate2Conditioning.ConcatChannels(latents, conditioning);
                Tensor vCond = _transformer.Forward(Backend, modelInput, promptEmbeds, tEmb, driving, referenceClipEmbeds);
                if (useCfg)
                {
                    // Strictly after the positive branch: the transformer's caches are unsynchronized.
                    Tensor vUncond = _transformer.Forward(Backend, modelInput, negativeEmbeds, tEmb, driving,
                        referenceClipEmbeds, unconditional: true);
                    LancePipelineCommon.CfgCombineRenormInPlace(vCond, vUncond, guidance, _config.CfgRescale);
                    vUncond.Dispose();
                }
                modelInput.Dispose();
                if (alternate) unipc!.Step(latents, vCond); else dpm!.Step(latents, vCond);
                vCond.Dispose();
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = latents,
                    LatentArch = LatentArchitecture.Wan,
                });
                Backend.FreeActivations();
                stream.EndStep();
                // Window the profiler onto the steady-state steps; step 0 carries the streaming window's warm-up.
                if (k == 0) Backend.ResetOpProfile();
            }
            Backend.DumpOpProfile($"animate2-denoise{Math.Max(1, steps - 1)}");
            Logs.Info($"Wan-Animate-2 denoise: {steps} steps in {phase.Elapsed.TotalMilliseconds / 1000.0:F1}s.");
        }
        catch
        {
            latents?.Dispose();
            latents = null;
            throw;
        }
        finally
        {
            // Before the sync: a forward that OOM'd mid-step surfaces on the next Sync, and a throw there would
            // leave the streaming hook attached on a transformer the recipe cache keeps alive.
            stream?.Dispose();
            Backend.Sync();
            driving?.Dispose();
            drivingLatent?.Dispose();
            conditioning?.Dispose();
        }
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor video = WanAnimate2Conditioning.TrimReferenceFrame(latents!);
        latents!.Dispose();
        Tensor rgb;
        Stopwatch decodeSw = Stopwatch.StartNew();
        Backend.ResetOpProfile();
        try { rgb = _vae.Decode(Backend, video); }
        finally { video.Dispose(); }
        Backend.DumpOpProfile("animate2-vaedecode");
        Logs.Info($"Wan-Animate-2 VAE decode: {decodeSw.Elapsed.TotalMilliseconds / 1000.0:F1}s.");
        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        Logs.Info($"Wan-Animate-2 chunk complete ({frames.Length} frames, seed={seed}).");
        return (frames, pixW, pixH, seed);
    }

    /// <summary>The generation stream's pixel clip: mid-grey everywhere (<c>torch.zeros</c> in [-1,1] space, which is
    /// grey and not black), with the carried frame written over frame 0 on a continuation chunk. The causal VAE
    /// leaks that frame forward, so the later latent frames are NOT the encode of grey alone — which is exactly why
    /// this is one encode rather than a splice.</summary>
    private static Tensor BuildContinuationPixels(Tensor? carriedRgbFrame, int pixT, int pixH, int pixW)
    {
        Tensor grey = new Tensor(new TensorShape([1L, 3, pixT, pixH, pixW]), DType.F32);
        new Span<float>((float*)grey.DataPointer, (int)grey.Shape.ElementCount).Clear();
        if (carriedRgbFrame is null)
        {
            return grey;
        }
        float* dst = (float*)grey.DataPointer;
        float* src = (float*)carriedRgbFrame.DataPointer;
        long frame = (long)pixH * pixW;
        for (int c = 0; c < 3; c++)
        {
            Buffer.MemoryCopy(src + (long)c * frame, dst + (long)c * pixT * frame, frame * sizeof(float), frame * sizeof(float));
        }
        return grey;
    }

    /// <summary>Fresh host-materialized copy of a (possibly device-resident) tensor — the form that survives the
    /// per-step activation sweeps and re-faults to device on use.</summary>
    private static Tensor HostCopy(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, x.DType);
        long bytes = x.DType.ComputeByteCount(x.ElementCount);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
        return o;
    }
}
