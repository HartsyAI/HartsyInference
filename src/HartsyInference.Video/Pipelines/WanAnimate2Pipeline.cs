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
/// fixed timestep to fill the per-block K/V cache, and the denoise loop splices that cache in frame-aligned.
/// There is no pose render, no face crop, no motion encoder and no retargeting — every V1 conditioning surface is
/// gone, and the driving pixels go straight to the VAE.
///
/// <para><b>CFG runs sequentially, deliberately.</b> <see cref="WanAnimate2Transformer"/>'s RoPE/text caches and its
/// log-scale bias are unsynchronized, so the two branches must not be dispatched to concurrent backends; this
/// pipeline never accepts a <see cref="DiffusionPipelineBase.CfgParallelBackend"/> and its recipe warns when one is
/// configured.</para>
///
/// <para><b>Numerics unvalidated.</b> No real-weight generation has been run against this yet, and the reference
/// sampler (FlowDPM++ 2M midpoint) is not ported — see <see cref="SubstitutedSampler"/>.</para></summary>
public sealed unsafe class WanAnimate2Pipeline : DiffusionPipelineBase
{
    /// <summary>The reference's clip length: 81 pixel frames, of which the last-but-one chunk boundary carries
    /// exactly one frame forward.</summary>
    public const int ClipLength = 81;

    /// <summary>Pixel frames of overlap between consecutive chunks (upstream's <c>first_num</c>).</summary>
    public const int ChunkOverlapFrames = 1;

    /// <summary>Upstream's <c>sample_shift</c>, applied to the sigmas.</summary>
    public const float DefaultFlowShift = 5f;

    /// <summary>The solver actually run. The reference uses FlowDPM++ 2M midpoint over
    /// <c>get_sampling_sigmas</c>; the engine has no flow-prediction DPM++, so UniPC stands in and every generation
    /// says so. This is a real numerical divergence, not a rename.</summary>
    public const string SubstitutedSampler = "unipc";

    private readonly WanAnimate2Transformer _transformer;
    private readonly IWanVaeDecoder _vae;
    private readonly IWanVaeEncoder _encoder;
    private readonly WanVideoConfig _config;

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

    /// <summary>True when the denoise must run the negative branch at all. At or below 1.0 the CFG fold is the
    /// identity, and the reference takes a single forward — which also means the block-9 uncond skip never fires
    /// for the distillation build.</summary>
    public static bool UsesCfgBranch(float guidance) => guidance > 1f;

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
    /// state of <b>this chunk's</b> driving frame 0, not the reference image's.</summary>
    public (byte[][] Frames, int Width, int Height, int Seed) GenerateChunk(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor drivingPromptEmbeds,
        Tensor referenceRgb, Tensor drivingRgbClip, TextToImageRequest request,
        Tensor? referenceClipEmbeds = null, Tensor? drivingClipEmbeds = null,
        Tensor? carriedRgbFrame = null, Action<GenerationProgress>? onProgress = null)
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

        Logs.Info($"Wan-Animate-2: {pixT}f {pixW}x{pixH}, {steps} steps, cfg={guidance}{(useCfg ? "" : " single forward")}, "
            + $"seed={seed} (gen latent {latentCh}x{tTotal}x{hLat}x{wLat}, driving {tTotal - 1} frame(s)"
            + $"{(continuation ? ", continuation chunk" : "")}, log_scale={_config.Animate2LogScale}).");
        Logs.Warning("[WanAnimate2] Sampling with UniPC. The reference solver is FlowDPM++ 2M midpoint over "
            + "get_sampling_sigmas, which this engine does not implement — the trajectory WILL differ from upstream, "
            + "and this output is not a numerical parity reference.");

        Tensor? conditioning = null, drivingLatent = null, latents = null;
        WanAnimate2DrivingCache? driving = null;
        BlockStreamingScope? stream = null;
        try
        {
            // One causal VAE encode per stream. The continuation frame goes over the HEAD of a mid-grey clip and is
            // encoded WITH it: the Wan VAE is temporally causal, so encoding the carried frame separately and
            // splicing latents is not the same tensor.
            Backend.PreloadWeights(_encoder.EnumerateWeights());
            Tensor referenceLatent = _encoder.Encode(Backend, referenceRgb);
            Tensor videoPixels = BuildContinuationPixels(carriedRgbFrame, pixT, pixH, pixW);
            Tensor videoLatent = _encoder.Encode(Backend, videoPixels);
            videoPixels.Dispose();
            Tensor drivingLatentDev = _encoder.Encode(Backend, drivingRgbClip);
            Backend.Sync();
            Backend.FreeWeights(_encoder.EnumerateWeights());

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
            long headroomBytes = Math.Max(EnvSwitch.GetLong("HARTSY_ANIMATE2_HEADROOM_MB", 3072) * 1024 * 1024,
                tokenLoad * ActivationBytesPerToken + FixedHeadroomBytes);
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
            driving = _transformer.EncodeDriving(Backend, drivingLatent, drivingPromptEmbeds, drivingClipEmbeds, genGrid);
            stream.EndStep();
            Backend.FreeActivations();

            latents = SeedGenerator.CreateNoise(new TensorShape([1L, latentCh, tTotal, hLat, wLat]), seed);
            FlowUniPCMultistepScheduler scheduler = new(solverOrder: 2);
            scheduler.SetTimesteps(steps, shift);

            for (int k = 0; k < steps; k++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                float tEmb = scheduler.Timesteps[k];
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
                scheduler.Step(latents, vCond);
                vCond.Dispose();
                sw.Stop();
                onProgress?.Invoke(new GenerationProgress(k + 1, steps, sw.Elapsed.TotalMilliseconds)
                {
                    Latent = latents,
                    LatentArch = LatentArchitecture.Wan,
                });
                Backend.FreeActivations();
                stream.EndStep();
            }
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
        try { rgb = _vae.Decode(Backend, video); }
        finally { video.Dispose(); }
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
