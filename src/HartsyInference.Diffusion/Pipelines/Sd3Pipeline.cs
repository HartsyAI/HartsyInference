using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>SD3 text-to-image and image-to-image pipeline. Orchestrates triple text encoding (CLIP-L + CLIP-G + T5-XXL) → MMDiT denoising with flow matching → VAE decode → RGB image output. T5 is optional for reduced VRAM usage.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>). Requires a <see cref="VaeEncoder"/> on the constructor. Setting <see cref="ImageToImageRequest.Mask"/> additionally enables blend-on-vanilla inpaint, identical to the SDXL/Flux paths but on SD3's 16-channel NCHW latent.</para></summary>
public sealed unsafe class Sd3Pipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly T5TextEncoder? _t5;
    private readonly Sd3Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly float _schedulerShift;

    /// <summary>Creates a new SD3 pipeline with all components pre-loaded. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public Sd3Pipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder? t5, Sd3Transformer transformer, VaeDecoder vaeDecoder,
        float schedulerShift = 3.0f)
        : this(backend, clipL, clipG, t5, transformer, vaeDecoder, vaeEncoder: null, schedulerShift)
    {
    }

    /// <summary>Creates a new SD3 pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public Sd3Pipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder? t5, Sd3Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, float schedulerShift = 3.0f)
        : base(backend)
    {
        _clipL = clipL;
        _clipG = clipG;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized input for all three text encoders.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int[] negativePromptTokenIdsL,
        int[] promptTokenIdsG,
        int[] negativePromptTokenIdsG,
        int promptEosPositionL,
        int negativeEosPositionL,
        int promptEosPositionG,
        int negativeEosPositionG,
        int[]? promptTokenIdsT5,
        int[]? negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Sd35.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string mode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                    : isImg2Img ? $"img2img (start={startStep}/{steps})"
                    : "txt2img";
        Logs.Info($"SD3 {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with triple encoders ─────────────────────────
        Logs.Info("Encoding text with CLIP-L, CLIP-G, T5...");

        // Bulk-upload T5 weights once (~5 GB on T5-XXL) so the encoder's many kernels don't
        // each pay a per-op cache-miss H2D transfer. Backends without a weight cache no-op.
        // Pairs with the FreeWeights at the END of this phase (not after the denoise loop). CLIP-L/G are
        // preloaded here too: "tiny" was wrong for CLIP-G, which is ~1.4 GB.
        if (_t5 is not null) Backend.PreloadWeights(_t5.EnumerateWeights());
        Backend.PreloadWeights(_clipL.EnumerateWeights());
        Backend.PreloadWeights(_clipG.EnumerateWeights());

        bool useCfg = cfgScale > 1.0f;

        // Encode positive prompt
        // SD3 is specified against the penultimate CLIP layer; request.ClipSkip overrides (2 = penultimate).
        int clipSkip = request.ClipSkip ?? 2;
        (Tensor condContext, Tensor condPooled) = EncodePrompt(
            promptTokenIdsL, promptTokenIdsG, promptTokenIdsT5,
            promptEosPositionL, promptEosPositionG,
            promptAttentionMaskT5, clipSkip);

        // Project context through transformer's context_embedder
        Tensor condProjected = _transformer.ProjectContext(Backend, condContext);
        condContext.Dispose();

        Tensor? uncondProjected = null;
        Tensor? uncondPooled = null;

        if (useCfg)
        {
            // Encode negative prompt
            (Tensor negContext, Tensor negPooled) = EncodePrompt(
                negativePromptTokenIdsL, negativePromptTokenIdsG, negativePromptTokenIdsT5,
                negativeEosPositionL, negativeEosPositionG,
                negativeAttentionMaskT5, clipSkip);

            uncondProjected = _transformer.ProjectContext(Backend, negContext);
            negContext.Dispose();
            uncondPooled = negPooled;
        }

        // Materialize the conditioning on the host, then reclaim every encoder intermediate. CLIP-L/G + T5
        // leave hundreds of device-cached activations that otherwise linger until GC finalization — they'd
        // hold multi-GB into the MMDiT phase. The pooled tensors are already host (ConcatPooled writes on the
        // CPU) but the ProjectContext outputs are live GPU activations; touching DataPointer forces the D2H
        // sync + cache eviction, making them safe to keep across the FreeActivations calls below.
        _ = condProjected.DataPointer;
        _ = condPooled.DataPointer;
        if (uncondProjected is not null) _ = uncondProjected.DataPointer;
        if (uncondPooled is not null) _ = uncondPooled.DataPointer;
        Backend.FreeActivations();

        // Release all three text encoders here, not after the denoise loop. They are not touched again this
        // generation (the conditioning above is already host-materialized), and holding them costs roughly
        // T5-XXL ~5 GB + CLIP-G ~1.4 GB + CLIP-L ~0.25 GB ≈ 6.6 GB of dead weight for the *entire* loop —
        // beside a ~5 GB transformer that is about to be preloaded. That is what made SD3.5-Medium OOM during
        // this very phase on a 12 GB card. Same staging every other pipeline uses (Lens frees its encoder at
        // the same boundary; Qwen evicts the DiT for the TE phase and re-preloads after).
        Backend.Sync();
        if (_t5 is not null) Backend.FreeWeights(_t5.EnumerateWeights());
        Backend.FreeWeights(_clipL.EnumerateWeights());
        Backend.FreeWeights(_clipG.EnumerateWeights());

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Set up flow-match scheduler ──────────────────────────────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial latent ─────────────────────────────────────
        // T2I: noise scaled by initSigma. Img2img: VAE-encode source + AddNoise at sigma[startStep].
        // Masked inpaint: keep clean source latent + downsampled mask alive for per-step blend.
        Tensor latent;
        Tensor? sourceLatent = null;
        Tensor? latentMask = null;
        if (request is ImageToImageRequest img2imgInit)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            sourceLatent = _vaeEncoder!.Encode(Backend, img2imgInit.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            using Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            latent = new Tensor(latentShape, DType.F32);
            AddFlowMatchNoise(Backend, scheduler, latent, sourceLatent, noise, startStep);

            if (isMaskedInpaint)
            {
                // Downsampling is setup-only host work. Stage its result through a backend op once so the same
                // device mask can be pinned and reused by every denoise step without repeated H2D uploads.
                using Tensor hostLatentMask =
                    MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
                latentMask = new Tensor(hostLatentMask.Shape, DType.F32);
                Backend.Scale(latentMask, hostLatentMask, 1.0f);
            }
            else
            {
                sourceLatent.Dispose();
                sourceLatent = null;
            }
        }
        else
        {
            latent = SeedGenerator.CreateNoise(latentShape, seed);
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = new Tensor(latentShape, DType.F32);
                Backend.Scale(scaled, latent, initSigma);
                latent.Dispose();
                latent = scaled;
            }
        }

        // ── 4. Denoising loop ───────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop. The transformer is touched
        // every step, every block — without preload, the first step would pay cache-miss
        // overhead for every parameter access. FreeWeights below the VAE handoff cycles the
        // VRAM. No-op on backends without a weight cache.
        // DiT sharding: asymmetric preload — Backend gets the shared weights PLUS its block range,
        // DitShardBackend gets ONLY its block range. Preloading EnumerateWeights() on both would replicate
        // the MMDiT instead of pooling VRAM across the two cards (the CFG-parallel mistake this exists to avoid).
        if (DitShardBackend is not null)
        {
            Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
        }

        // Across-step First-Block cache (docs/Research/STEP_ACCELERATION.md §2; same knobs as ZImagePipeline).
        // Excluded from img2img/inpaint (block-0 indicator drift semantics haven't been validated against a
        // partially-noised init latent) and DiT sharding (ForwardSharded has no cache-consuming entry point —
        // combining the two is out of scope here). SD3 runs true CFG with two independent forward passes per
        // step, so — unlike ZImagePipeline's fastPath, which only ever runs cache-free CFG — this needs one
        // DeviceFeatureCache PER STREAM (their hidden states differ, per the type's own doc).
        bool stepCacheFastPath = !isImg2Img && !isMaskedInpaint && DitShardBackend is null;
        (float stepCacheThreshold, int stepCacheCap, float[]? stepCachePoly, float stepCacheLate) = StepCacheEnv.Resolve(null);
        DeviceFeatureCache? stepCacheCond = null;
        DeviceFeatureCache? stepCacheUncond = null;
        if (stepCacheThreshold > 0f && stepCacheFastPath)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                stepCacheCond = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                stepCacheUncond = useCfg ? new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile()) : null;
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

        Logs.Info("Starting SD3 denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        long denoiseD2hStart = Backend.GetD2hSyncCount();
        bool latentPinned = false;
        bool sourcePinned = false;
        bool maskPinned = false;
        try
        {
            // Preserve all loop-carried state on device while reclaiming VAE-encode intermediates. Pinning a host
            // tensor is harmless on CPU and also covers t2i's initially-host noise: once its first device buffer is
            // planted, the pre-existing pin protects it. Img2img latent/source and the staged mask are resident now.
            Backend.PinActivation(latent);
            latentPinned = true;
            if (sourceLatent is not null)
            {
                Backend.PinActivation(sourceLatent);
                sourcePinned = true;
            }
            if (latentMask is not null)
            {
                Backend.PinActivation(latentMask);
                maskPinned = true;
            }
            Backend.FreeActivations();

            for (int i = startStep; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                float t = timesteps[i];

                // Late-window gate: reuse eligible only in the schedule tail (mirrors ZImagePipeline).
                bool cacheEligible = stepCacheLate <= 0f || (i + 1) > steps * (1f - stepCacheLate);

                // Flow-match Euler is exactly z += dt * velocity. Fold standard CFG and the Euler update into
                // one in-place backend op. Final modulation and unpatchify now keep both predictions and latent
                // device-resident. A progress consumer may intentionally read the latent; otherwise both ordinary
                // and masked-inpaint loops remain device-resident.
                if (useCfg)
                {
                    Tensor? uncondNoise = null;
                    Tensor? condNoise = null;
                    try
                    {
                        // Preserve the established branch order: the transformer can own request-local caches.
                        uncondNoise = RunForward(
                            latent, t, uncondProjected!, uncondPooled!,
                            cacheEligible ? stepCacheUncond : null);
                        condNoise = RunForward(
                            latent, t, condProjected, condPooled,
                            cacheEligible ? stepCacheCond : null);
                        Backend.CfgEulerStep(latent, condNoise, uncondNoise, cfgScale, scheduler.Dt(i));
                    }
                    finally
                    {
                        condNoise?.Dispose();
                        uncondNoise?.Dispose();
                    }
                }
                else
                {
                    Tensor noisePred = RunForward(
                        latent, t, condProjected, condPooled,
                        cacheEligible ? stepCacheCond : null);
                    try
                    {
                        Backend.CfgEulerStep(latent, noisePred, noisePred, 1.0f, scheduler.Dt(i));
                    }
                    finally
                    {
                        noisePred.Dispose();
                    }
                }

                // A preview consumer may intentionally read the latent and thereby remove its pin. Re-asserting
                // the pin after every device update makes the ordinary no-preview path survive FreeActivations;
                // a deliberate read remains correct because it first materializes the current device contents.
                Backend.PinActivation(latent);

                // Masked-inpaint blend: replace the unmasked region with the source's trajectory at next sigma.
                // The backend fuses source re-noising and dense channel-broadcast blending into this one in-place
                // launch. The final step supplies noise=null and sigma=0, selecting the clean source exactly.
                if (latentMask is not null && sourceLatent is not null)
                {
                    BlendMaskedSourceTrajectory(
                        Backend, scheduler, latent, sourceLatent, latentMask, seed, i + 1);
                }

                stepSw.Stop();
                Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.Sd3,
                });

                // Deterministically reclaim dead MMDiT intermediates without discarding the pinned latent.
                // Keep the async pool reservation hot across steps; the final phase sweep trims it once.
                Backend.FreeActivations(trimPool: false);
            }
        }
        finally
        {
            if (maskPinned) Backend.UnpinActivation(latentMask!);
            if (sourcePinned) Backend.UnpinActivation(sourceLatent!);
            if (latentPinned) Backend.UnpinActivation(latent);
        }
        Logs.Debug($"SD3 denoise-loop D2H syncs: {Backend.GetD2hSyncCount() - denoiseD2hStart}.");

        // DecodeTiled may attempt a full-frame decode, sweep activation memory after an OOM, then fall back to
        // host-sliced tiles. Materialize the one final latent here so that recovery sweep cannot discard the
        // only authoritative (device) copy and make the fallback decode stale pre-denoise host data.
        _ = latent.DataPointer;

        if (stepCacheCond is not null)
        {
            string uncondStats = stepCacheUncond is not null
                ? $"; uncond {stepCacheUncond.Computes} computes / {stepCacheUncond.Reuses} reuses"
                : "";
            Logs.Info($"Step cache: cond {stepCacheCond.Computes} computes / {stepCacheCond.Reuses} reuses{uncondStats}");
        }

        condProjected.Dispose();
        condPooled.Dispose();
        uncondProjected?.Dispose();
        uncondPooled?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();
        stepCacheCond?.Dispose();
        stepCacheUncond?.Dispose();

        Sd3Transformer.DumpFinalLatent(latent);

        // Free transformer + text encoder weights from GPU now that denoising is done.
        // The VAE im2col buffers at 512×512+ are large (hundreds of MB to several GB),
        // and on a 12 GB card we OOM during VAE decode if the transformer is still
        // resident. Mirrors PHASE_3_DEVIATIONS #18 (UNet eviction before VAE for SDXL/Flux).
        // Backends without a weight cache (CPU/Vulkan) treat this as a no-op.
        Backend.Sync();
        DitShardBackend?.Sync();
        if (DitShardBackend is not null)
        {
            // Mirror of the sharded preload above: Backend frees shared + its block range, DitShardBackend
            // frees ONLY its block range. Freeing EnumerateWeights() on Backend alone would ask it to free
            // tensors it never promoted (they're on DitShardBackend) and leak DitShardBackend's range across
            // generations.
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
        }
        else
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
        }
        // T5/CLIP are already gone — released at the end of the text-encode phase rather than held across
        // the whole denoise loop.

        // ── 5. VAE decode ───────────────────────────────────────────────
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile. Internal fast-path
        // skips tiling when the latent fits in a single tile, so small images pay no overhead.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 6. Pixel-space recomposite for masked inpaint ──────────────
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            RecomposeMaskedImage(
                Backend, image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // ── 7. Convert to RGB bytes ─────────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the budget of whatever generation runs next.
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"SD3 {mode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Device-routable equivalent of <see cref="FlowMatchEulerDiscreteScheduler.AddNoise"/>.</summary>
    internal static void AddFlowMatchNoise(
        IBackend backend,
        FlowMatchEulerDiscreteScheduler scheduler,
        Tensor output,
        Tensor source,
        Tensor noise,
        int stepIndex)
    {
        float sigma = scheduler.SigmaAt(stepIndex);
        backend.AffineMix(output, source, noise, 1.0f - sigma, sigma);
    }

    /// <summary>
    /// Keeps an SD3 inpaint latent's unmasked region on the source trajectory at <paramref name="nextStep"/>.
    /// Nonterminal steps reproduce the established <c>seed + nextStep</c> fresh-noise sequence; the terminal step
    /// passes a null noise tensor with zero noise scale and therefore blends the clean source.
    /// </summary>
    internal static void BlendMaskedSourceTrajectory(
        IBackend backend,
        FlowMatchEulerDiscreteScheduler scheduler,
        Tensor target,
        Tensor source,
        Tensor mask,
        int seed,
        int nextStep)
    {
        float sigma = scheduler.SigmaAt(nextStep);
        Tensor? freshNoise = null;
        try
        {
            if (nextStep < scheduler.NumInferenceSteps)
                freshNoise = SeedGenerator.CreateNoise(target.Shape, seed + nextStep);
            backend.MaskedAffineMixInPlace(
                target, source, freshNoise, mask,
                sourceScale: 1.0f - sigma,
                noiseScale: sigma,
                layout: MaskBroadcastLayout.DenseNchwBroadcast);
        }
        finally
        {
            freshNoise?.Dispose();
        }
    }

    /// <summary>Device-routable final pixel recomposite; mask=1 selects the decoded image.</summary>
    internal static void RecomposeMaskedImage(IBackend backend, Tensor image, Tensor sourceImage, Tensor pixelMask)
    {
        backend.MaskedAffineMixInPlace(
            image, sourceImage, noise: null, mask: pixelMask,
            sourceScale: 1.0f,
            noiseScale: 0.0f,
            layout: MaskBroadcastLayout.DenseNchwBroadcast);
    }

    /// <summary>Encodes a single prompt through all three text encoders and combines the results.</summary>
    private (Tensor context, Tensor pooled) EncodePrompt(
        int[] tokenIdsL, int[] tokenIdsG, int[]? tokenIdsT5,
        int eosPositionL, int eosPositionG,
        int[]? attentionMaskT5, int clipSkip = 2)
    {
        int seqLenClip = tokenIdsL.Length;

        // CLIP-L: penultimate hidden [1, 77, 768] + pooled [1, 768]
        int[][] batchTokenIdsL = [tokenIdsL];
        int[] eosPositionsL = [eosPositionL];
        (Tensor clipLHidden, Tensor? clipLPooled) = _clipL.EncodePenultimate(Backend, batchTokenIdsL, eosPositionsL, clipSkip);

        // CLIP-G: penultimate hidden [1, 77, 1280] + pooled [1, 1280]
        int[][] batchTokenIdsG = [tokenIdsG];
        int[] eosPositionsG = [eosPositionG];
        (Tensor clipGHidden, Tensor? clipGPooled) = _clipG.EncodePenultimate(Backend, batchTokenIdsG, eosPositionsG, clipSkip);

        // Combine pooled: concat(clip_l_pooled, clip_g_pooled, dim=-1) → [1, 2048]
        Tensor pooled = ConcatPooled(clipLPooled!, clipGPooled!);
        clipLPooled?.Dispose();
        clipGPooled?.Dispose();

        // Combine hidden: concat(clip_l_hidden, clip_g_hidden, dim=-1) → [1, 77, 2048]
        Tensor lgHidden = CfgHelper.ConcatLastDim(clipLHidden, clipGHidden);
        clipLHidden.Dispose();
        clipGHidden.Dispose();

        // Pad to 4096: [1, 77, 2048] → [1, 77, 4096]
        int lgDim = (int)lgHidden.Shape[2];
        int targetDim = 4096;
        Tensor lgPadded = PadLastDim(lgHidden, lgDim, targetDim);
        lgHidden.Dispose();

        // T5 encoding
        Tensor t5Hidden;
        if (_t5 is not null && tokenIdsT5 is not null)
        {
            int[][] batchT5 = [tokenIdsT5];
            int[][]? batchMask = attentionMaskT5 is not null ? [attentionMaskT5] : null;
            t5Hidden = _t5.Encode(Backend, batchT5, batchMask);
        }
        else
        {
            // T5 dropout: zero tensor [1, seqLen, 4096]
            int t5SeqLen = tokenIdsT5?.Length ?? seqLenClip;
            TensorShape t5Shape = new TensorShape(1, t5SeqLen, targetDim);
            t5Hidden = new Tensor(t5Shape, DType.F32);
            Backend.Fill(t5Hidden, 0.0f);
        }

        // Concat along sequence: [1, 77, 4096] + [1, 77, 4096] → [1, 154, 4096]
        Tensor context = ConcatAlongSeqDim(lgPadded, t5Hidden);
        lgPadded.Dispose();
        t5Hidden.Dispose();

        return (context, pooled);
    }

    /// <summary>Routes one denoise step through <see cref="DitShardBackend"/>'s block-range split when
    /// configured, else the normal single-backend path. <paramref name="stepCache"/> only applies on the
    /// non-sharded path — <see cref="Sd3Transformer.ForwardSharded"/> has no cache-consuming entry point.</summary>
    private Tensor RunForward(Tensor latent, float timestep, Tensor context, Tensor pooled, Utilities.DeviceFeatureCache? stepCache = null)
    {
        if (DitShardBackend is not null)
        {
            return _transformer.ForwardSharded(Backend, DitShardBackend, latent, timestep, context, pooled, DitShardSplitBlock);
        }
        return _transformer.Forward(Backend, latent, timestep, context, pooled, stepCache);
    }

    /// <summary>Concatenates two [B, S1, D] and [B, S2, D] tensors along the sequence dimension → [B, S1+S2, D].</summary>
    private static Tensor ConcatAlongSeqDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqA = (int)a.Shape[1];
        int seqB = (int)b.Shape[1];
        int dim = (int)a.Shape[2];
        int seqOut = seqA + seqB;

        TensorShape outShape = new TensorShape(batch, seqOut, dim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            // Copy all of tensor a for this batch
            int aSrcOffset = bIdx * seqA * dim;
            int aDstOffset = bIdx * seqOut * dim;
            Buffer.MemoryCopy(aPtr + aSrcOffset, outPtr + aDstOffset, seqA * dim * sizeof(float), seqA * dim * sizeof(float));

            // Copy all of tensor b for this batch
            int bSrcOffset = bIdx * seqB * dim;
            int bDstOffset = bIdx * seqOut * dim + seqA * dim;
            Buffer.MemoryCopy(bPtr + bSrcOffset, outPtr + bDstOffset, seqB * dim * sizeof(float), seqB * dim * sizeof(float));
        }

        return output;
    }

    /// <summary>Pads the last dimension with zeros: [B, S, currentDim] → [B, S, targetDim].</summary>
    private static Tensor PadLastDim(Tensor input, int currentDim, int targetDim)
    {
        if (currentDim == targetDim)
            return input;

        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];

        TensorShape outShape = new TensorShape(batch, seqLen, targetDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Zero the entire output first, then copy input data
        int totalElements = batch * seqLen * targetDim;
        for (int i = 0; i < totalElements; i++)
            outPtr[i] = 0.0f;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * currentDim;
                int outOffset = (b * seqLen + s) * targetDim;
                Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, currentDim * sizeof(float), currentDim * sizeof(float));
            }
        }

        return output;
    }

    /// <summary>Concatenates two pooled tensors [B, D1] and [B, D2] along the last dimension → [B, D1+D2].</summary>
    private static Tensor ConcatPooled(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int dimA = (int)a.Shape[1];
        int dimB = (int)b.Shape[1];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            Buffer.MemoryCopy(aPtr + bIdx * dimA, outPtr + bIdx * dimOut, dimA * sizeof(float), dimA * sizeof(float));
            Buffer.MemoryCopy(bPtr + bIdx * dimB, outPtr + bIdx * dimOut + dimA, dimB * sizeof(float), dimB * sizeof(float));
        }

        return output;
    }
}
