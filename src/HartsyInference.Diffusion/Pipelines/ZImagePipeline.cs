using HartsyInference.Diffusion.Sampling;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Z-Image text-to-image and image-to-image pipeline (Tongyi Lab, Apache 2.0). Accepts pre-computed Qwen3-4B caption embeddings (text-encoder forward is owned by a separate component) and orchestrates the Lumina2/NextDiT transformer with a static-shift flow-match Euler scheduler.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromEmbeddings"/>. Requires a <see cref="VaeEncoder"/> on construction. Strength=0 byte-identical pass-through is exact; nonzero-strength img2img produces structurally-correct output but quality has not been validated against a Python reference. For tight refining quality on Z-Image, prefer the cross-model pixel-space pattern (run a different pipeline as the refiner). Latent normalization is handled in one place — <see cref="VaeDecoder"/>'s internal UndoScaling — same as Flux.</para>
/// </summary>
public sealed unsafe class ZImagePipeline : DiffusionPipelineBase
{
    private readonly ZImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly ZImageConfig _config;

    /// <summary>Keeps lazily promoted DiT weights and fused-attention plans resident across generations. A prompt-cache miss still evicts them before a same-device Qwen encode because the 8 GB encoder and 6.2 GB Z-Image transformer cannot coexist on a 12 GB card.</summary>
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);

    /// <summary>Explicit bring-up probe. Each scan deliberately materializes a prediction on the host, so it is disabled in production; use it to localize non-finite Base outputs before the CFG combine.</summary>
    private static readonly bool PredictionStatsEnabled = EngineKnobs.ZimagePredStats.Value;

    private GenerationDefaults VariantDefaults =>
        _config.IsBase ? GenerationDefaults.ZImageBase : GenerationDefaults.ZImageTurbo;

    /// <summary>Creates a Z-Image pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public ZImagePipeline(IBackend backend, ZImageTransformer transformer, VaeDecoder vaeDecoder, ZImageConfig config)
        : this(backend, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Z-Image pipeline with both VAE halves loaded. Required for img2img.</summary>
    public ZImagePipeline(IBackend backend, ZImageTransformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, ZImageConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen3 caption embeddings. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    ///   <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial latent = fresh Gaussian noise scaled by initSigma; denoise from step 0).</item>
    ///   <item><see cref="ImageToImageRequest"/> → image-to-image. The source image is encoded via the 16-channel Flux/Z-Image VAE and combined with fresh noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Requires a <see cref="VaeEncoder"/>.</item>
    /// </list></summary>
    /// <param name="captionEmbeddings">Last-hidden-state output of Qwen3-4B for the prompt [B, capLen, 2560]. The Z-Image system prompt + chat template should already be applied upstream.</param>
    /// <param name="request">Generation parameters (Width, Height, Steps, Seed). Pass an <see cref="ImageToImageRequest"/> for img2img.</param>
    /// <param name="cfgScale">Classifier-free guidance scale. Use 1.0 for Turbo (no CFG, single forward per step). Use 3.0–5.0 for Base when a negative-prompt embedding is also provided.</param>
    /// <param name="negativeCaptionEmbeddings">Optional negative-prompt embeddings for CFG. Required when <paramref name="cfgScale"/> &gt; 1.0.</param>
    /// <param name="onProgress">Optional progress callback per step.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeCaptionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null,
        RegionalPlan? regionalPlan = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        ValidateRequest(captionEmbeddings, request, cfgScale, negativeCaptionEmbeddings);
        try
        {
            DeviceFeatureCache? stepCache = null;
            Exception? generationError = null;
            try
            {
                try
                {
                    return GenerateFromEmbeddingsCore(captionEmbeddings, request, cfgScale,
                        negativeCaptionEmbeddings, onProgress, regionalPlan, ref stepCache);
                }
                catch (Exception ex)
                {
                    generationError = ex;
                    throw;
                }
            }
            finally
            {
                // The cache pins its snapshots/residuals so per-step activation sweeps cannot reclaim them.
                // Dispose before the outer failure rollback; otherwise cancellation leaves those pins alive. If
                // generation already failed, cache cleanup is best-effort so it cannot replace that exception.
                if (stepCache is not null)
                {
                    if (generationError is null)
                        stepCache.Dispose();
                    else
                        TryCleanup("step cache", stepCache.Dispose);
                }
            }
        }
        catch
        {
            // Generation can escape through cancellation/progress callbacks and validation/runtime failures, not
            // only OutOfVramException (the engine's global cleanup boundary). Leave this cached pipeline reusable:
            // discard only its own phase state and execution plans, never the backend's unrelated model weights.
            CleanupFailedGeneration();
            throw;
        }
    }

    private void ValidateRequest(Tensor captionEmbeddings, TextToImageRequest request, float cfgScale,
        Tensor? negativeCaptionEmbeddings)
    {
        ArgumentNullException.ThrowIfNull(captionEmbeddings);
        ArgumentNullException.ThrowIfNull(request);
        if (!float.IsFinite(cfgScale))
            throw new ArgumentOutOfRangeException(nameof(cfgScale), cfgScale, "Z-Image CFG scale must be finite.");
        if (request is ImageToImageRequest && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");
        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0 (Z-Image-Base path). For Z-Image-Turbo, leave cfgScale at 1.0.",
                nameof(negativeCaptionEmbeddings));

        GenerationDefaults defaults = VariantDefaults;
        int width = request.Width ?? defaults.Width;
        int height = request.Height ?? defaults.Height;
        int steps = request.Steps ?? defaults.Steps;
        _ = Img2ImgSetup.Prepare(request, height, width, steps);
    }

    private (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddingsCore(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale,
        Tensor? negativeCaptionEmbeddings,
        Action<GenerationProgress>? onProgress,
        RegionalPlan? regionalPlan,
        ref DeviceFeatureCache? stepCacheInst)
    {
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0 (Z-Image-Base path). For Z-Image-Turbo, leave cfgScale at 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        GenerationDefaults defaults = VariantDefaults;
        int width = request.Width ?? defaults.Width;
        int height = request.Height ?? defaults.Height;
        int latentH = height / _config.VaeDownscaleFactor;
        int latentW = width / _config.VaeDownscaleFactor;
        int steps = request.Steps ?? defaults.Steps;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        // A cache-hit generation skips the wrapper's Qwen phase, so the prior DiT may still be resident. The
        // same-device VAE encoder cannot safely preload beside it on a 12 GB card; make this phase boundary just as
        // explicit as a prompt-cache miss. Strength=0 returned above and pays no needless eviction.
        if (isImg2Img && ReferenceEquals(VaeBackend, Backend))
            EvictResidentWeights();

        string opMode = isMaskedInpaint ? $"inpaint (start={plan.StartStep}/{steps})"
                      : isImg2Img ? $"img2img (start={plan.StartStep}/{steps})"
                      : "txt2img";
        Logs.Info($"Z-Image {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Static-shift flow-match Euler scheduler ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Build initial latent (t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at sigma[startStep]) ──
        //    For masked inpaint the clean source latent is kept alive for per-step blending.
        TensorOwnership tensors = new();
        try
        {
            (Tensor initialLatent, Tensor? initialSourceLatent) = BuildInitialLatent(
                request, scheduler, latentShape, seed, plan.StartStep, keepSourceLatent: isMaskedInpaint);
            Tensor latent = tensors.Own(initialLatent, "latent");
            Tensor? sourceLatent = initialSourceLatent is null ? null
                : tensors.Own(initialSourceLatent, "source latent");
            Tensor? latentMask = null;
            if (isMaskedInpaint)
            {
                latentMask = tensors.Own(
                    MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW),
                    "latent mask");
            }

            // Materialize every tensor that must survive across steps on the host, then reclaim the VAE-encode
            // intermediates. The caption embeddings arrive from an upstream Qwen3 encode and may still be live GPU
            // activations; the t2i initSigma path leaves `latent` as a live Backend.Scale output. The per-step
            // FreeActivations below frees device buffers WITHOUT a D2H sync-back, so anything still device-only
            // here would be silently lost — touching DataPointer forces the D2H sync + cache eviction.
            _ = captionEmbeddings.DataPointer;
            if (negativeCaptionEmbeddings is not null)
                _ = negativeCaptionEmbeddings.DataPointer;
            _ = latent.DataPointer;
            if (sourceLatent is not null)
                _ = sourceLatent.DataPointer;
            if (latentMask is not null)
                _ = latentMask.DataPointer;
            if (regionalPlan is not null)
            {
                // Region conditioning is handed to the transformer every step; it too may be a live GPU activation.
                foreach (RegionConditioning region in regionalPlan.Regions)
                    _ = region.Cond.DataPointer;
            }
            Backend.FreeActivations();

            // ── 3. Denoising loop (from startStep onward) ──
            // Unmasked, non-regional t2i/img2img keeps the latent in packed token space across the WHOLE loop,
            // including Z-Image-Base CFG. Patchify once, run cond/uncond via ForwardPacked, then fold the model's
            // mandatory velocity negation, non-standard cond-anchored CFG, and Euler update into one device op.
            // No prediction or latent is read through DataPointer per step. Inpaint/regional still require the
            // pixel-space path below for source-trajectory blending / spatial conditioning.
            bool useCfg = cfgScale > 1.0f;
            bool fastPath = CanUsePackedDenoise(isMaskedInpaint, regionalPlan);
            if (fastPath)
            {
                long captionPreparationD2hStart = Backend.GetD2hSyncCount();
                _ = _transformer.PreparePackedCaption(Backend, captionEmbeddings);
                if (useCfg)
                    _ = _transformer.PreparePackedCaption(Backend, negativeCaptionEmbeddings!);
                long captionPreparationD2h = Backend.GetD2hSyncCount() - captionPreparationD2hStart;
                Logs.Debug($"Z-Image packed-caption preparation D2H syncs: {captionPreparationD2h}.");
            }
            long denoiseD2hStart = Backend.GetD2hSyncCount();
            // Default-off across-step First-Block cache (HARTSY_STEP_CACHE / _LATE — fleet knobs, wired on the
            // packed t2i path only; source-conditioned img2img needs its own quality calibration before this
            // approximate reuse is admitted. No calibrated profile yet: "=1" resolves to the generic raw 0.10
            // until the Z-Image A/B lands. Armed cache forces the eager path (no graph).
            (float stepCacheThreshold, int stepCacheCap, float[]? stepCachePoly, float stepCacheLate) =
                StepCacheEnv.Resolve(null);
            if (stepCacheThreshold > 0f && CanUseStepCache(fastPath, isImg2Img, useCfg))
            {
                if (Backend.SupportsDeviceStepCacheGate)
                {
                    stepCacheInst = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                    Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}"
                        + (stepCachePoly is not null ? ", poly gate" : "")
                        + (stepCacheLate > 0f ? $", lateWindow={stepCacheLate}" : ""));
                }
                else
                {
                    Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                        "(stepcache.ptx not compiled?) — running uncached.");
                }
            }
            // Step-graph mode (HARTSY_DIT_GRAPH, fast path only): route the latent through the transformer's FIXED
            // buffer so the captured graph's baked address stays valid across steps and gens. The fixed tensor is
            // transformer-owned: never disposed here, never DataPointer-read (snapshot instead).
            bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);
            bool graphMode = fastPath && !useCfg && stepCacheInst is null && !nonDefaultSampler
                && Models.Denoisers.DiTBlocks.DitStepGraph.Enabled && Backend.StepGraphSupported;
            int fpH = latentH / _config.PatchSize;
            int fpW = latentW / _config.PatchSize;
            Tensor? packed = null;
            if (fastPath)
            {
                long phP = sw.ElapsedMilliseconds;
                Tensor fresh = tensors.Own(_transformer.PatchifyLatent(Backend, latent), "fresh packed latent");
                tensors.DisposeOwned(latent);
                if (graphMode)
                {
                    packed = _transformer.PrepareGraphLatent(Backend, fresh);
                    tensors.DisposeOwned(fresh);
                }
                else
                {
                    packed = fresh;
                }
                Logs.Verbose($"[zimage-phase] patchify={sw.ElapsedMilliseconds - phP}ms");
            }

            ReadOnlySpan<float> timesteps = scheduler.Timesteps;

            // Sampler selection (2026-08-20). Z-Image is NextDiT: it predicts −v and folds diffusers' mandatory
            // noise_pred = −noise_pred into the step delta, which is why the predictor reports
            // PredictionType.NegatedFlowVelocity — the sampler flips a scalar coefficient instead of negating a
            // latent-sized tensor on every forward.
            ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Z-Image",
                startsFromNoisedInit: plan.StartStep > 0);
            if (!fastPath && nonDefaultSampler)
            {
                throw new NotSupportedException(
                    $"Sampler/schedule '{request.Scheduler}' runs only on Z-Image's packed fast path, and this "
                    + "generation fell back to the reference loop (masked inpaint or regional prompting). Drop the "
                    + "sampler selection, or drop the feature that forced the fallback.");
            }
            float[] timestepTable = timesteps.ToArray();
            // `stepCacheInst` is a `ref` parameter of this method and cannot be captured by a lambda; the cache
            // instance itself is what the closure needs, and it is never reassigned after this point.
            DeviceFeatureCache? stepCache = stepCacheInst;
            int startStep = plan.StartStep;

            for (int i = plan.StartStep; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                float sigma = timesteps[i] / 1000.0f;

                // Diffusers Z-Image pipeline inverts the timestep: it feeds the transformer
                // `(1 - sigma)` (then the transformer multiplies by t_scale=1000 internally). Without
                // this inversion every step conditions on the OPPOSITE point in the schedule and the
                // model produces near-random output. See pipeline_z_image.py:506.
                float invertedSigma = 1.0f - sigma;

                if (fastPath)
                {
                    DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                        PredictionType.NegatedFlowVelocity,
                        (x, s, stepIndex) =>
                        {
                            // Diffusers' Z-Image pipeline inverts the timestep: the transformer is fed (1 − sigma).
                            // On-schedule sigmas reuse the loop's own expression — the F32 round trip through x1000
                            // is not exact, so raw sigma would shift every existing generation by an ulp.
                            float stepSigma = stepIndex < steps && s == scheduler.SigmaAt(stepIndex)
                                ? timestepTable[stepIndex] / 1000.0f : s;
                            float inverted = 1.0f - stepSigma;
                            // Narrowed for a non-default sampler: the step cache's drift calibration assumes one
                            // forward per step, which a second-order method breaks.
                            bool eligible = !nonDefaultSampler
                                && (stepCacheLate <= 0f || (stepIndex + 1) > steps * (1f - stepCacheLate));
                            Tensor cond = _transformer.ForwardPacked(Backend, x, captionEmbeddings, inverted, fpH, fpW,
                                eligible ? stepCache : null);
                            if (PredictionStatsEnabled)
                                ValidatePredictionFinite(cond, $"conditional step {stepIndex + 1}", logStats: true);
                            if (!useCfg)
                            {
                                // Graph mode borrows the transformer's fixed velocity buffer, which must not be freed.
                                return new DenoisePrediction(cond, cond, ownsTensors: !graphMode);
                            }
                            Tensor uncond = _transformer.ForwardPacked(Backend, x, negativeCaptionEmbeddings!, inverted, fpH, fpW);
                            if (PredictionStatsEnabled)
                                ValidatePredictionFinite(uncond, $"unconditional step {stepIndex + 1}", logStats: true);
                            // CfgEulerStep computes g·cond + (1−g)·uncond. Z-Image requires
                            // cond + cfg·(cond − uncond) = (1+cfg)·cond − cfg·uncond, hence g = cfg + 1.
                            return new DenoisePrediction(cond, uncond, cfgScale + 1.0f);
                        });
                    if (i == startStep)
                    {
                        sampler.Reset(packed!.Shape);
                    }
                    sampler.Step(Backend, packed!, predictor, i);
                    stepSw.Stop();
                    Logs.Verbose($"[zimage-phase] step {i + 1}/{steps}: {stepSw.ElapsedMilliseconds}ms");
                    // No Latent in the progress event: the packed tokens must never be DataPointer-read mid-loop
                    // (the lazy D2H would free the device buffer). Previews skip gracefully when Latent is null.
                    onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
                    continue;
                }

                Tensor velocity = tensors.Own(
                    _transformer.Forward(Backend, latent, captionEmbeddings, invertedSigma, regionalPlan, i - plan.StartStep),
                    "conditional velocity");

                if (cfgScale > 1.0f)
                {
                    Tensor uncondVelocity = tensors.Own(
                        _transformer.Forward(Backend, latent, negativeCaptionEmbeddings!, invertedSigma),
                        "unconditional velocity");
                    Tensor combined = tensors.Own(ApplyZImageCfg(velocity, uncondVelocity, cfgScale), "CFG velocity");
                    tensors.DisposeOwned(uncondVelocity);
                    tensors.DisposeOwned(velocity);
                    velocity = combined;
                }

                // Diffusers Z-Image pipeline does `noise_pred = -noise_pred` (see pipeline_z_image.py:558).
                // Empirically required: without this we get pure RGB noise; with it we get structured output.
                NegateInPlace(velocity);

                Tensor newLatent = tensors.Own(new Tensor(latentShape, DType.F32), "next latent");
                scheduler.Step(newLatent, velocity, latent, i);
                tensors.DisposeOwned(velocity);
                tensors.DisposeOwned(latent);
                latent = newLatent;

                // Masked-inpaint blend: keep unmasked region on the source's flow-matching trajectory
                // by re-noising the source latent at the next step's sigma. Final step blends with the
                // clean source — no further denoising follows.
                if (latentMask is not null && sourceLatent is not null)
                {
                    int nextStep = i + 1;
                    Tensor noisedSource;
                    if (nextStep < steps)
                    {
                        Tensor freshNoise = tensors.Own(
                            SeedGenerator.CreateNoise(latentShape, seed + nextStep),
                            "inpaint noise");
                        noisedSource = tensors.Own(new Tensor(latentShape, DType.F32), "noised source latent");
                        scheduler.AddNoise(noisedSource, sourceLatent, freshNoise, nextStep);
                        tensors.DisposeOwned(freshNoise);
                    }
                    else
                    {
                        noisedSource = sourceLatent;
                    }
                    MaskBlendUtilities.BlendChannelsInPlace(latent, noisedSource, latentMask);
                    if (noisedSource != sourceLatent)
                        tensors.DisposeOwned(noisedSource);
                }

                stepSw.Stop();
                Logs.Debug($"Z-Image step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.ZImage,
                });

                // Reclaim GPU-resident activation buffers between steps: the DiT keeps intermediates on-device and
                // any not read-back/disposed linger until GC, accumulating to OOM over the schedule (same fix as
                // Flux / the video pipelines). Safe: scheduler.Step runs on the host (and NegateInPlace already
                // synced the velocity), so `latent` — the only tensor the next step needs — is host-resident, and
                // everything else persistent was materialized above the loop.
                // trimPool:false — the next step re-uses the pool reservation directly; trimming here cost a full
                // stream-sync + multi-GB driver release/re-map EVERY step (the final FreeActivations below trims).
                Backend.FreeActivations(trimPool: false);
            }

            // Fast path: bring the final packed tokens back to pixel space once for the VAE — on-device
            // (backend.UnpatchifyTokens), so the loop → VAE chain never leaves the GPU: the host unpatchify loop
            // D2H-drained the tokens (~280 ms: pipeline drain + 4M-element triple loop) and the VAE re-uploaded the
            // result. Graph mode reads a SNAPSHOT — touching the fixed buffer directly would free what the captured
            // graph points at.
            if (fastPath)
            {
                long phU = sw.ElapsedMilliseconds;
                Tensor tokens = graphMode ? _transformer.SnapshotGraphLatent(Backend) : packed!;
                if (graphMode)
                    tensors.Own(tokens, "graph latent snapshot");
                latent = tensors.Own(new Tensor(latentShape, DType.F32), "unpatchified latent");
                Backend.UnpatchifyTokens(latent, tokens, _config.InChannels, fpH, fpW, _config.PatchSize,
                    innerChannelFastest: true);
                tensors.DisposeOwned(tokens);   // graph mode: the snapshot; eager: packed itself (as before)
                Logs.Verbose($"[zimage-phase] unpatchify={sw.ElapsedMilliseconds - phU}ms");
            }

            Logs.Debug($"Z-Image denoise-loop D2H syncs: {Backend.GetD2hSyncCount() - denoiseD2hStart}.");

            // Base has historically collapsed to an all-NaN latent on its first denoise step. One scalar device
            // reduction after the complete loop catches any non-finite value before an expensive VAE decode while
            // preserving per-step residency. Turbo skips this gate to keep its established output/perf path exact.
            if (_config.IsBase)
                ValidateFiniteTensor(Backend, latent, "final Base latent");

            if (stepCacheInst is not null)
            {
                Logs.Info($"Step cache: {stepCacheInst.Computes} computes / {stepCacheInst.Reuses} reuses");
            }

            // HARTSY_KEEP_MODELS controls only the expensive DiT residency. The current Z-Image path deliberately
            // keeps its proven lazy/auto-promotion behavior rather than forcing a 6.2 GB bulk preload on 12 GB cards;
            // this exact free still removes every promoted weight and cached dtype cast when residency is disabled.
            // Invalidate the step graph before freeing because its captured kernels bake those weight addresses.
            if (!KeepModelsResident)
                EvictResidentWeights();

            // ── 4. VAE decode ──
            // No pre-scale: VaeDecoder.UndoScaling already applies `latent / ScalingFactor + ShiftFactor`
            // using VaeConfig.ZImage (== VaeConfig.Flux: scale=0.3611, shift=0.1159). The previous
            // `PrepareVaeInput` step did the same arithmetic in the pipeline and then DecodeTiled
            // ran it again inside UndoScaling — double-scaling pushed the latent ~2.77× too high
            // and saturated every conv layer. Fix: hand the raw latent directly to DecodeTiled
            // and trust the single normalize-on-decode contract that Flux's pipeline already
            // follows. DecodeTiled caps im2col workspace at ~2.4 GB per tile and fast-paths to
            // a single direct decode when the latent fits in one tile.
            //
            // Diagnostic: log per-channel min/max/mean of the latent BEFORE the VAE sees it.
            // Compare against Flux's healthy distribution (typically min~-4, max~+4, mean<|2|)
            // — if Z-Image's are wildly different (saturated, out-of-range, all near 0) the
            // bug isn't in the VAE, it's upstream in the denoise loop or scheduler.
            LogLatentStatsPerChannel("Pre-VAE latent", latent);

            // Placement boundary: a VAE on another device cannot consume the primary backend's cached activation.
            // Host-materialize once; same-device runs retain the existing device-resident handoff and numerical path.
            if (!ReferenceEquals(VaeBackend, Backend))
                _ = latent.DataPointer;

            Logs.Verbose("Decoding latents to image (tiled F32 path)...");
            Stopwatch vaeSw = Stopwatch.StartNew();
            VaeBackend.PreloadWeights(_vaeDecoder.EnumerateWeights());
            Tensor image = tensors.Own(_vaeDecoder.DecodeTiled(VaeBackend, latent), "decoded image");
            tensors.DisposeOwned(latent);
            if (sourceLatent is not null)
                tensors.DisposeOwned(sourceLatent);
            if (latentMask is not null)
                tensors.DisposeOwned(latentMask);
            vaeSw.Stop();
            Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

            LogLatentStatsPerChannel("VAE output", image);

            // ── 5. Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
            //    Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
            if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
            {
                MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
            }

            // ── 6. RGB conversion — device CHW F32 → HWC u8, one 3 MB D2H (see Krea2Pipeline) ──
            long phRgb = sw.ElapsedMilliseconds;
            byte[] rgbData;
            {
                int outH = (int)image.Shape[2], outW = (int)image.Shape[3];
                Tensor hwcU8 = tensors.Own(
                    new Tensor(new TensorShape(outH, outW, 3), DType.U8),
                    "RGB staging tensor");
                VaeBackend.ChwF32ToHwcU8(hwcU8, image);
                rgbData = new byte[(long)outH * outW * 3];
                unsafe
                {
                    fixed (byte* dst = rgbData)
                        Buffer.MemoryCopy((void*)hwcU8.DataPointer, dst, rgbData.Length, rgbData.Length);
                }
                tensors.DisposeOwned(hwcU8);
            }
            // A uniformly black/white frame is an error only when the decode path itself collapsed (NaN/Inf) —
            // a legitimate prompt or an inpaint over a solid source can genuinely produce one.
            string? collapseEndpoint = DetectEndpointCollapse(rgbData);
            if (collapseEndpoint is not null)
            {
                ValidateFiniteTensor(VaeBackend, image, $"decoded frame (uniformly {collapseEndpoint})");
                Logs.Warning($"[Z-Image] decoded frame is uniformly {collapseEndpoint} but the decode path is " +
                    "finite — accepting as a legitimate solid-color image.");
            }
            Logs.Verbose($"[zimage-phase] rgb={sw.ElapsedMilliseconds - phRgb}ms");
            tensors.DisposeOwned(image);

            // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
            // memory until GC finalization and shrink the budget of whatever generation runs next.
            VaeBackend.FreeActivations();
            if (!ReferenceEquals(VaeBackend, Backend))
                Backend.FreeActivations();

            // FreeActivations just freed the step graph's fixed boundary buffers (they live in the ACTIVATION
            // cache — CopyInto/CfgEulerStep re-cache them there), so a cross-gen replay would launch against freed
            // addresses: CUDA 700 that poisons the context (hit live on the first warm gen). The graph is therefore
            // per-generation here: invalidate now, re-warm + re-capture next gen at the fresh buffer addresses.
            if (graphMode)
                _transformer.InvalidateStepGraph(Backend);

            sw.Stop();
            Logs.Info($"Z-Image {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

            tensors.EnsureEmpty();
            return (rgbData, width, height, seed);
        }
        catch (Exception generationError)
        {
            // All tensors registered here are local generation owners. Graph-mode packed/velocity buffers are
            // intentionally absent because the transformer owns those fixed-address capture boundaries.
            tensors.DisposeAllAfterFailure(generationError);
            throw;
        }
    }

    /// <summary>Whether the denoise loop can remain in packed token space. The latent's origin is deliberately irrelevant: a plain img2img latent has already completed VAE encode + flow-noise mixing before this decision. Masks require per-step source blending in pixel latent space; regional prompts require the regional forward.</summary>
    internal static bool CanUsePackedDenoise(bool isMaskedInpaint, RegionalPlan? regionalPlan) =>
        !isMaskedInpaint && (regionalPlan is null || regionalPlan.Regions.Count == 0);

    /// <summary>Across-step reuse is currently calibrated only for single-pass packed t2i. Img2img starts from a source-conditioned trajectory and remains excluded until a dedicated quality A/B establishes safe gates.</summary>
    internal static bool CanUseStepCache(bool packedDenoise, bool isImg2Img, bool useCfg) =>
        packedDenoise && !isImg2Img && !useCfg;

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: VaeEncoder.Encode(source) combined with fresh noise via flow-matching AddNoise at sigma[startStep].
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean source latent is returned alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request, FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep, bool keepSourceLatent)
    {
        TensorOwnership tensors = new();
        try
        {
            if (request is ImageToImageRequest img2img)
            {
                Stopwatch vaeEncSw = Stopwatch.StartNew();
                Tensor? sourceLatent = null;
                Exception? encodeError = null;
                try
                {
                    VaeBackend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
                    sourceLatent = tensors.Own(
                        _vaeEncoder.Encode(VaeBackend, img2img.SourceImage),
                        "encoded source latent");
                    // The scheduler below is host-side, and the DiT may live on another device. Make the phase
                    // boundary explicit before reclaiming encoder activations and handing the latent onward.
                    _ = sourceLatent.DataPointer;
                }
                catch (Exception ex)
                {
                    encodeError = ex;
                    throw;
                }
                finally
                {
                    Exception? cleanupError = null;
                    try
                    {
                        VaeBackend.FreeWeights(_vaeEncoder!.EnumerateWeights());
                    }
                    catch (Exception ex)
                    {
                        cleanupError = ex;
                    }
                    try
                    {
                        VaeBackend.FreeActivations();
                    }
                    catch (Exception ex)
                    {
                        cleanupError ??= ex;
                    }
                    if (cleanupError is not null)
                    {
                        if (encodeError is null)
                        {
                            throw new InvalidOperationException(
                                "Z-Image VAE encoding completed, but its device phase could not be reclaimed.", cleanupError);
                        }
                        Logs.Warning($"[Z-Image] VAE-encoder cleanup also failed while propagating " +
                            $"'{encodeError.Message}': {cleanupError.Message}");
                    }
                }
                vaeEncSw.Stop();
                Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

                Tensor noise = tensors.Own(TakeOrCreateNoise(request, latentShape, seed), "img2img noise");
                Tensor latent = tensors.Own(new Tensor(latentShape, DType.F32), "initial img2img latent");
                scheduler.AddNoise(latent, sourceLatent!, noise, startStep);
                tensors.DisposeOwned(noise);
                if (keepSourceLatent)
                {
                    return (tensors.Transfer(latent), tensors.Transfer(sourceLatent!));
                }
                tensors.DisposeOwned(sourceLatent!);
                return (tensors.Transfer(latent), null);
            }

            Tensor t2iNoise = tensors.Own(TakeOrCreateNoise(request, latentShape, seed), "txt2img noise");
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = tensors.Own(new Tensor(latentShape, DType.F32), "scaled txt2img latent");
                Backend.Scale(scaled, t2iNoise, initSigma);
                tensors.DisposeOwned(t2iNoise);
                return (tensors.Transfer(scaled), null);
            }
            return (tensors.Transfer(t2iNoise), null);
        }
        catch (Exception generationError)
        {
            tensors.DisposeAllAfterFailure(generationError);
            throw;
        }
    }

    /// <summary>Releases only Z-Image's resident denoiser state before a same-device text-encoder phase or when <c>HARTSY_KEEP_MODELS=0</c>. The model intentionally remains lazily promoted; this method is the exact teardown mirror and also invalidates graph/attention plans that contain device addresses.</summary>
    public void EvictResidentWeights()
    {
        Exception? firstError = null;
        Try(Backend.Sync);
        Try(() => _transformer.InvalidateStepGraph(Backend));
        Try(() => _transformer.ReleaseDeviceCache(Backend));
        Try(() => Backend.FreeWeights(_transformer.EnumerateWeights()));
        Try(Backend.ReleaseAttentionExecutionCache);
        if (firstError is not null)
            throw new InvalidOperationException("Z-Image could not fully reclaim its denoiser device state.", firstError);

        void Try(Action cleanup)
        {
            try
            { cleanup(); }
            catch (Exception error) { firstError ??= error; }
        }
    }

    /// <summary>Reclaims this recipe's VAE device state before a text encoder is staged on the same device. Successful generations deliberately keep the decoder warm, but an 8 GB Qwen preload takes priority at this explicit phase boundary. Every independent owner is attempted before an error is reported.</summary>
    public void EvictVaeDeviceState()
    {
        Exception? firstError = null;
        Try(VaeBackend.Sync);
        Try(() => VaeBackend.FreeWeights(_vaeDecoder.EnumerateWeights()));
        if (_vaeEncoder is not null)
            Try(() => VaeBackend.FreeWeights(_vaeEncoder.EnumerateWeights()));
        Try(() => VaeBackend.ReleaseAttentionExecutionCache());
        Try(() => VaeBackend.FreeActivations(trimPool: true));
        if (firstError is not null)
            throw new InvalidOperationException("Z-Image could not fully reclaim its VAE device state.", firstError);

        void Try(Action cleanup)
        {
            try
            { cleanup(); }
            catch (Exception error) { firstError ??= error; }
        }
    }

    /// <summary>Best-effort targeted rollback for an interrupted generation. Do not let cleanup replace the cancellation/runtime exception the caller needs to diagnose, and do not sweep unrelated backend weights.</summary>
    private void CleanupFailedGeneration()
    {
        TryCleanup("DiT state", EvictResidentWeights);
        TryCleanup("VAE decoder weights", () => VaeBackend.FreeWeights(_vaeDecoder.EnumerateWeights()));
        if (_vaeEncoder is not null)
            TryCleanup("VAE encoder weights", () => VaeBackend.FreeWeights(_vaeEncoder.EnumerateWeights()));
        TryCleanup("primary activations", () => Backend.FreeActivations());
        if (!ReferenceEquals(VaeBackend, Backend))
        {
            TryCleanup("VAE attention plans", () => VaeBackend.ReleaseAttentionExecutionCache());
            TryCleanup("VAE activations", () => VaeBackend.FreeActivations());
        }
    }

    private static void TryCleanup(string phase, Action cleanup)
    {
        try
        {
            cleanup();
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Z-Image] Failed to release {phase} during rollback: {ex.Message}");
        }
    }

    /// <summary>Releases device-side copies owned by this pipeline. Injected component objects remain caller-owned per <see cref="DiffusionPipelineBase"/>; <c>ZImageRecipePipeline</c> disposes those host owners afterwards.</summary>
    protected override void DisposeCore() => CleanupFailedGeneration();

    /// <summary>Returns the endpoint label ("black (0)" / "white (255)") when the entire RGB frame sits at a single byte endpoint, else null. Endpoint collapse alone is NOT proof of failure — a legitimate prompt or an inpaint over a solid source can genuinely produce one — so the caller disambiguates by checking the decoded F32 tensor for NaN/Inf before rejecting.</summary>
    internal static string? DetectEndpointCollapse(ReadOnlySpan<byte> rgb)
    {
        if (rgb.IsEmpty)
            throw new InvalidOperationException("Z-Image produced an empty RGB frame.");

        bool anyNonBlack = false;
        bool anyNonWhite = false;
        foreach (byte value in rgb)
        {
            anyNonBlack |= value != 0;
            anyNonWhite |= value != byte.MaxValue;
            if (anyNonBlack && anyNonWhite)
                return null;
        }
        return anyNonBlack ? "white (255)" : "black (0)";
    }

    /// <summary>Rejects a tensor containing NaN/Inf. Device backends with a resident reduction read back only the scalar result; other backends use the diagnostic host scan.</summary>
    internal static void ValidateFiniteTensor(IBackend backend, Tensor tensor, string label)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tensor);
        if (backend.SupportsDeviceStepCacheGate)
        {
            float selfDistance = backend.RelativeL1Distance(tensor, tensor);
            if (float.IsFinite(selfDistance))
                return;
            throw new InvalidOperationException($"Z-Image {label} contains NaN/Inf or overflowed values.");
        }
        ValidatePredictionFinite(tensor, label, logStats: false);
    }

    /// <summary>Host diagnostic used by HARTSY_ZIMAGE_PRED_STATS=1. This intentionally forces a D2H sync and must not be enabled for performance measurements.</summary>
    internal static void ValidatePredictionFinite(Tensor tensor, string label, bool logStats)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.DType != DType.F32)
            throw new NotSupportedException($"Z-Image prediction diagnostics require F32, got {tensor.DType}.");

        float* values = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        long bad = 0;
        double sum = 0.0, sumSquares = 0.0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        for (long i = 0; i < count; i++)
        {
            float value = values[i];
            if (!float.IsFinite(value))
            {
                bad++;
                continue;
            }
            min = MathF.Min(min, value);
            max = MathF.Max(max, value);
            sum += value;
            sumSquares += (double)value * value;
        }
        long finite = count - bad;
        double mean = finite > 0 ? sum / finite : double.NaN;
        double variance = finite > 0 ? Math.Max(0.0, sumSquares / finite - mean * mean) : double.NaN;
        if (logStats)
        {
            Logs.Info($"[zimage-pred] {label}: shape={tensor.Shape} min={min:G6} max={max:G6} " +
                $"mean={mean:G6} std={Math.Sqrt(variance):G6} nonfinite={bad}/{count}");
        }
        if (bad != 0)
            throw new InvalidOperationException(
                $"Z-Image {label} contains {bad}/{count} NaN/Inf values before CFG/Euler combine.");
    }

    /// <summary>In-place negate. Z-Image's diffusers pipeline negates the velocity output before stepping.</summary>
    private static void NegateInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long count = t.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            p[i] = -p[i];
    }

    /// <summary>Z-Image CFG combine: <c>combined = cond + cfg * (cond - uncond)</c> = <c>(1+cfg)*cond - cfg*uncond</c>. NON-STANDARD: the conventional formula is <c>uncond + cfg * (cond - uncond)</c> = <c>(1-cfg)*uncond + cfg*cond</c>; Z-Image's diffusers pipeline (<c>pipeline_z_image.py:541</c>) uses cond as the baseline and amplifies the (cond - uncond) direction. At cfg=4.0 this gives <c>5*cond - 4*uncond</c>, vs the standard <c>4*cond - 3*uncond</c> — a meaningfully different signal. <see cref="CfgHelper.ApplyCfg"/> is therefore NOT a drop-in replacement; this stays local.</summary>
    private static Tensor ApplyZImageCfg(Tensor cond, Tensor uncond, float cfg)
    {
        if (cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new ArgumentException($"ApplyZImageCfg requires F32 inputs; got cond={cond.DType}, uncond={uncond.DType}.");
        if (!cond.Shape.Equals(uncond.Shape))
            throw new ArgumentException($"ApplyZImageCfg shape mismatch: cond={cond.Shape}, uncond={uncond.Shape}.");
        TensorOwnership tensors = new();
        try
        {
            Tensor output = tensors.Own(new Tensor(cond.Shape, DType.F32), "CFG output");
            float* condPtr = (float*)cond.DataPointer;
            float* uncondPtr = (float*)uncond.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            long count = output.Shape.ElementCount;
            for (long i = 0; i < count; i++)
            {
                float c = condPtr[i];
                outPtr[i] = c + cfg * (c - uncondPtr[i]);
            }
            return tensors.Transfer(output);
        }
        catch (Exception cfgError)
        {
            tensors.DisposeAllAfterFailure(cfgError);
            throw;
        }
    }

    /// <summary>Tracks only tensors created by the current generation. Borrowed inputs and transformer-owned graph buffers are never registered. Removing an owner before disposal prevents the failure sweep from double-disposing a tensor whose normal teardown itself threw.</summary>
    private sealed class TensorOwnership
    {
        private readonly List<(Tensor Tensor, string Name)> _owned = [];

        public Tensor Own(Tensor tensor, string name)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            if (Find(tensor) >= 0)
                throw new InvalidOperationException($"Tensor '{name}' was registered twice with one generation.");
            _owned.Add((tensor, name));
            return tensor;
        }

        public Tensor Transfer(Tensor tensor)
        {
            Remove(tensor);
            return tensor;
        }

        public void DisposeOwned(Tensor tensor)
        {
            Remove(tensor);
            tensor.Dispose();
        }

        public void EnsureEmpty()
        {
            if (_owned.Count != 0)
                throw new InvalidOperationException(
                    $"Z-Image generation completed with {_owned.Count} untransferred tensor owner(s).");
        }

        public void DisposeAllAfterFailure(Exception primaryError)
        {
            for (int i = _owned.Count - 1; i >= 0; i--)
            {
                (Tensor tensor, string name) = _owned[i];
                _owned.RemoveAt(i);
                try
                {
                    tensor.Dispose();
                }
                catch (Exception cleanupError)
                {
                    Logs.Warning($"[Z-Image] Failed to dispose {name} while propagating " +
                        $"'{primaryError.Message}': {cleanupError.Message}");
                }
            }
        }

        private void Remove(Tensor tensor)
        {
            int index = Find(tensor);
            if (index < 0)
                throw new InvalidOperationException("Attempted to release a tensor not owned by this generation.");
            _owned.RemoveAt(index);
        }

        private int Find(Tensor tensor)
        {
            for (int i = _owned.Count - 1; i >= 0; i--)
            {
                if (ReferenceEquals(_owned[i].Tensor, tensor))
                    return i;
            }
            return -1;
        }
    }

    /// <summary>Per-channel min/max/mean diagnostic for a 4D NCHW tensor at Verbose level. Used to bracket the pre/post VAE state when tracking down all-black output bugs — healthy Z-Image / Flux latents have per-channel min ~-5 to -1, max ~+1 to +5, mean within ±2. RGB outputs should land in roughly [-1, 1] with mean near 0. Outside those bands means the model or VAE saturated.</summary>
    /// <summary>Diagnostic gate for the per-channel latent/VAE stats (HARTSY_ZIMAGE_STATS=1). Unconditional stats forced a D2H drain + a host scan of the full tensors every generation — pure overhead outside bring-up.</summary>
    private static readonly bool LatentStatsEnabled = EngineKnobs.ZimageStats.Value;

    private static void LogLatentStatsPerChannel(string name, Tensor t)
    {
        if (!LatentStatsEnabled)
            return;
        if (t.Shape.Rank != 4)
            return;
        int channels = (int)t.Shape[1];
        int spatial = (int)(t.Shape[2] * t.Shape[3]);
        float* ptr = (float*)t.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float min = float.MaxValue, max = float.MinValue, sum = 0;
            int nan = 0, inf = 0;
            for (int i = 0; i < spatial; i++)
            {
                float v = ptr[c * spatial + i];
                if (float.IsNaN(v))
                { nan++; continue; }
                if (float.IsInfinity(v))
                { inf++; continue; }
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
                sum += v;
            }
            float mean = spatial > 0 ? sum / spatial : 0;
            string flags = nan > 0 || inf > 0 ? $" nan={nan} inf={inf}" : "";
            Logs.Verbose($"  [{name}] ch{c}: min={min:F4} max={max:F4} mean={mean:F4}{flags}");
        }
    }
}
