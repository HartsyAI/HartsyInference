using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>SDXL text-to-image and image-to-image pipeline. Orchestrates dual CLIP text encoding (CLIP-L + CLIP-G) → UNet denoising with ADM conditioning → VAE decode → RGB image output.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromTokens"/> — same method, single API. Requires a <see cref="VaeEncoder"/> on construction. This also unlocks the cross-model refining pattern: any base pipeline's RGB output can be fed through this pipeline as a refiner via <see cref="ImagePostProcessor.RgbBytesToTensor"/> + img2img with low <c>Strength</c>.</para>
/// </summary>
public sealed class SdxlPipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly float _vaeScalingFactor;

    /// <summary>Standard-profile residency (HARTSY_KEEP_MODELS): UNet weights stay GPU-resident across generations, skipping the per-generation free + ~2 s re-upload. SDXL's UNet (2.5 GB F16) + BF16 VAE + dual CLIP fit 24 GB together, so no evict-for-TE dance is needed.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _unetResident;

    // Prompt-embedding cache: repeat prompts skip the whole dual-CLIP phase (~4.4 s/gen measured at
    // 1024²). Keyed on all four token streams + CLIP-G EOS positions + clip-skip. The cached tensors
    // are host-materialized at store time so a backend FreeActivations can't leave them stale.
    private int[]? _teKeyL, _teKeyLNeg, _teKeyG, _teKeyGNeg;
    private int _teKeyEosG = -1, _teKeyNegEosG = -1, _teKeyClipSkip = -1;
    private Tensor? _cachedTextEmb;
    private Tensor? _cachedPooled;

    /// <summary>Creates a new SDXL pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, float vaeScalingFactor = 0.13025f)
        : this(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder: null, vaeScalingFactor)
    {
    }

    /// <summary>Creates a new SDXL pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, float vaeScalingFactor = 0.13025f)
        : base(backend)
    {
        _clipL = clipL;
        _clipG = clipG;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _vaeScalingFactor = vaeScalingFactor;
    }

    /// <summary>Generates an image from pre-tokenized dual-CLIP input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial latent = noise * initSigma, denoise from step 0).</item>
    /// <item><see cref="ImageToImageRequest"/> → image-to-image (initial latent = VAE-encoded source + noise * sigma[startStep], denoise from <c>startStep = steps - round(steps * Strength)</c>). Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// The two paths share the entire dual-CLIP encoding, ADM conditioning, denoise loop, and VAE decode pipeline. Strength=0 short-circuits to byte-identical pass-through.
    /// </summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int[] negativePromptTokenIdsL,
        int[] promptTokenIdsG,
        int[] negativePromptTokenIdsG,
        int promptEosPositionG,
        int negativeEosPositionG,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        IReadOnlyList<ControlNetConditioning>? controlNets = null,
        RefinerSwapConfig? refiner = null,
        IReadOnlyList<IpAdapterConditioning>? ipAdapters = null,
        ConditioningSchedule? conditioningSchedule = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Sdxl.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        // Refiner StepSwap setup. Compute swap step from refiner Strength (fraction of total
        // steps the refiner runs at the END). Disabled when Strength<=0 or refiner is null.
        // For img2img / inpaint, the swap is still measured against the full schedule —
        // matches Comfy semantics so a user typing "Strength=0.2" gets the same swap point
        // regardless of img2img InitImageCreativity.
        int swapStep = -1;
        bool useStepSwap = false;
        float[]? refinerSizeConditionPos = null;
        float[]? refinerSizeConditionNeg = null;
        if (refiner is not null)
        {
            float refinerStrength = Math.Clamp(refiner.Strength, 0f, 1f);
            if (refinerStrength > 0f)
            {
                int refinerSteps = (int)MathF.Round(steps * refinerStrength);
                swapStep = Math.Clamp(steps - refinerSteps, startStep, steps);
                useStepSwap = swapStep < steps;
                refinerSizeConditionPos = [height, width, 0f, 0f, refiner.AestheticScore];
                refinerSizeConditionNeg = [height, width, 0f, 0f, refiner.NegativeAestheticScore];
            }
        }

        string mode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                    : isImg2Img ? $"img2img (start={startStep}/{steps})"
                    : "txt2img";
        if (useStepSwap) mode += $" + refiner@{swapStep}/{steps}";
        Logs.Info($"SDXL {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Dual CLIP text encoding (shared by both paths), with a cross-generation prompt-embedding
        //    cache. The refiner StepSwap path needs the raw CLIP-G hidden separately, so it bypasses the
        //    cache (rare path; the cache targets the repeat-prompt benchmark/iteration workflow).
        // SDXL is specified against the penultimate CLIP layer; request.ClipSkip overrides (2 = penultimate).
        int clipSkip = request.ClipSkip ?? 2;
        bool teCacheEligible = !useStepSwap;
        bool teCacheHit = teCacheEligible && _cachedTextEmb is not null && _cachedPooled is not null
            && _teKeyClipSkip == clipSkip
            && _teKeyEosG == promptEosPositionG && _teKeyNegEosG == negativeEosPositionG
            && TokensEqual(_teKeyL, promptTokenIdsL) && TokensEqual(_teKeyLNeg, negativePromptTokenIdsL)
            && TokensEqual(_teKeyG, promptTokenIdsG) && TokensEqual(_teKeyGNeg, negativePromptTokenIdsG);

        Tensor textEmbeddings;
        Tensor? pooledOutput;
        Tensor? clipGForRefiner = null;
        if (teCacheHit)
        {
            textEmbeddings = _cachedTextEmb!;
            pooledOutput = _cachedPooled;
            Logs.Info("SDXL prompt-embedding cache hit — text encoding skipped");
        }
        else
        {
            Logs.Info("Encoding text with dual CLIP encoders...");
            // Bulk-upload both encoders for the duration of the encode, then release them below (the
            // PreloadWeights/FreeWeights symmetry AGENTS.md requires). Only reached on a prompt-cache MISS —
            // a hit skips the whole phase, so repeat prompts pay nothing.
            TextEncoderBackend.PreloadWeights(_clipL.EnumerateWeights());
            TextEncoderBackend.PreloadWeights(_clipG.EnumerateWeights());
            int[][] batchTokenIdsL = [negativePromptTokenIdsL, promptTokenIdsL];
            (Tensor clipLHidden, _) = _clipL.EncodePenultimate(TextEncoderBackend, batchTokenIdsL, [0, 0], clipSkip);

            int[][] batchTokenIdsG = [negativePromptTokenIdsG, promptTokenIdsG];
            int[] eosPositions = [negativeEosPositionG, promptEosPositionG];
            (Tensor clipGHidden, Tensor? pooled) = _clipG.EncodePenultimate(TextEncoderBackend, batchTokenIdsG, eosPositions, clipSkip);
            pooledOutput = pooled;

            textEmbeddings = CfgHelper.ConcatLastDim(clipLHidden, clipGHidden);
            clipLHidden.Dispose();
            // Refiner phase cross-attends to CLIP-G alone (CrossAttentionDim=1280, not the 2048 concat). Prefer a
            // separate <refiner>-prompt conditioning when supplied, else reuse the base prompt's CLIP-G. clipGHidden
            // is disposed here unless it IS the chosen refiner conditioning (kept alive to the end of the pipeline).
            clipGForRefiner = useStepSwap ? (refiner!.RefinerConditioning ?? clipGHidden) : null;
            if (!ReferenceEquals(clipGForRefiner, clipGHidden)) clipGHidden.Dispose();

            // Release both encoders now the hidden states are extracted — they are not touched again in this
            // pipeline. CLIP-G alone is ~1.4 GB, so on a 12 GB card this is ~13% of the budget, and it is
            // budget the VAE decode actively bids for: VaeDecoder picks full-res vs tiled on measured free VRAM
            // (needs workspace + 1.5 GB headroom), so holding dead weights here can push a decode to the slower
            // tiled path for no reason. Sync first so the encode's in-flight reads finish before the free.
            TextEncoderBackend.Sync();
            TextEncoderBackend.FreeWeights(_clipL.EnumerateWeights());
            TextEncoderBackend.FreeWeights(_clipG.EnumerateWeights());
            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

            if (teCacheEligible && pooledOutput is not null)
            {
                _cachedTextEmb?.Dispose();
                _cachedPooled?.Dispose();
                // Host-materialize before caching (ConcatLastDim already built textEmbeddings on the host;
                // the pooled output is a live GPU activation until touched).
                unsafe { _ = (nint)pooledOutput.DataPointer; }
                _cachedTextEmb = textEmbeddings;
                _cachedPooled = pooledOutput;
                _teKeyL = (int[])promptTokenIdsL.Clone();
                _teKeyLNeg = (int[])negativePromptTokenIdsL.Clone();
                _teKeyG = (int[])promptTokenIdsG.Clone();
                _teKeyGNeg = (int[])negativePromptTokenIdsG.Clone();
                _teKeyEosG = promptEosPositionG;
                _teKeyNegEosG = negativeEosPositionG;
                _teKeyClipSkip = clipSkip;
            }
        }

        // 2. ADM size conditioning (orig_size = target_size = request resolution; no crop)
        float[] sizeCondition =
        [
            height, width,
            0f, 0f,
            height, width,
        ];

        // 3. Set up scheduler
        IScheduler scheduler = SchedulerFactory.Create(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 4. Build initial latent — t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at startStep.
        //    For masked inpaint we keep the clean source latent alive for per-step blending.
        (Tensor latent, Tensor? sourceLatent) = BuildInitialLatent(request, scheduler, latentShape, seed, startStep);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // 5. Denoise loop (both paths run the same loop from their respective startStep)
        // Bulk-upload UNet weights before the denoise loop. SDXL UNet is ~2.5 GB at F16 —
        // without preload the first step would pay cache-miss overhead for every parameter.
        // Under HARTSY_KEEP_MODELS the weights stay resident across generations and the
        // preload is skipped. No-op on backends without a weight cache.
        Stopwatch preloadSw = Stopwatch.StartNew();
        if (!_unetResident)
        {
            Backend.PreloadWeights(_unet.EnumerateWeights());
        }
        Logs.Verbose($"[sdxl-phase] UNet preload {(_unetResident ? "0 (resident)" : preloadSw.ElapsedMilliseconds.ToString())}ms");
        bool useF16 = (_unet.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;

        // Fused drain-free loop: batched-CFG single UNet forward per step + in-place device CFG-Euler
        // update. Applies to the plain t2i/img2img case on an epsilon-prediction Euler schedule; every
        // conditional feature (masked inpaint, ControlNet, IP-Adapter, refiner StepSwap, per-step
        // conditioning schedules, other schedulers) keeps the reference host loop.
        bool fusedLoop = scheduler is EulerDiscreteScheduler fusedScheduler && fusedScheduler.FusedEulerCompatible
            && !isMaskedInpaint
            && !useStepSwap
            && (controlNets is null || controlNets.Count == 0)
            && (ipAdapters is null || ipAdapters.Count == 0)
            && conditioningSchedule is null
            && pooledOutput is not null;
        Stopwatch denoiseSw = Stopwatch.StartNew();
        latent = fusedLoop
            ? RunDenoiseLoopFused(latent, latentShape, textEmbeddings, pooledOutput!, sizeCondition, (EulerDiscreteScheduler)scheduler, useF16, startStep, steps, cfgScale, onProgress)
            : RunDenoiseLoop(latent, latentShape, textEmbeddings, pooledOutput!, sizeCondition, scheduler, useF16, startStep, totalSteps: steps, cfgScale, sourceLatent, latentMask, seed, controlNets,
                refiner, swapStep, clipGForRefiner, refinerSizeConditionPos, refinerSizeConditionNeg, ipAdapters, onProgress, conditioningSchedule);
        Logs.Verbose($"[sdxl-phase] denoise {denoiseSw.ElapsedMilliseconds}ms ({(fusedLoop ? "fused" : "host")} loop)");

        if (!ReferenceEquals(textEmbeddings, _cachedTextEmb)) textEmbeddings.Dispose();
        if (pooledOutput is not null && !ReferenceEquals(pooledOutput, _cachedPooled)) pooledOutput.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();
        // Only dispose clipGForRefiner when it's the base clipGHidden we kept alive — a caller-supplied
        // RefinerConditioning is owned by the caller.
        if (clipGForRefiner is not null && !ReferenceEquals(clipGForRefiner, refiner?.RefinerConditioning))
        {
            clipGForRefiner.Dispose();
        }

        // 6. VAE decode. Under HARTSY_KEEP_MODELS the UNet stays resident (2.5 GB F16 beside the
        // VAE's banded-conv workspace fits 24 GB with ~14 GB headroom — measured peak 10.2 GB);
        // otherwise free it to reclaim VRAM for the high-res VAE conv2d buffers. CLIP-L/CLIP-G were
        // already released at the end of the text-encode phase above.
        Backend.Sync();
        if (KeepModelsResident)
        {
            _unetResident = true;
        }
        else
        {
            Backend.FreeWeights(_unet.EnumerateWeights());
            _unetResident = false;
        }

        // Pass the F32 denoise latent straight to DecodeTiled — it matches dtype to the VAE weights
        // internally (per-tile for the tiled path, once for the single-tile path). We deliberately do
        // NOT pre-cast the whole latent to BF16 here: at 1024² (128×128 latent → 3×3 tiles) that
        // full-tensor F32→BF16→F32 round-trip through the churned post-gen CUDA mem pool came back
        // partially written — only the first tile's region was valid, the rest decoded to NaN → a
        // mostly-black image. The F32 latent's host buffer is already valid, so tiled slicing is clean.
        Logs.Verbose("Decoding latents to image (F32 latent, dtype matched per-tile)...");
        Stopwatch vaeSw = Stopwatch.StartNew();

        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Tensor image = _vaeDecoder.DecodeTiled(VaeBackend, latent);
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 7. Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        //    Suppresses VAE encode/decode drift in unmasked regions, which would otherwise
        //    accumulate across repeated inpaint operations.
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // 8. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SDXL {mode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the starting latent for the denoise loop. T2i: fresh noise scaled by <c>InitialNoiseSigma</c>. Img2img: VAE-encoded source plus fresh noise injected at <c>sigma[startStep]</c> via <c>scheduler.AddNoise</c>.
    /// <para>Returns the clean source latent alongside the noised initial latent for img2img — masked inpaint reuses it per step. Caller disposes both. <c>sourceLatent</c> is null for txt2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request, IScheduler scheduler, TensorShape latentShape, int seed, int startStep)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(VaeBackend, img2img.SourceImage);  // LOAD-BEARING for VaeDevice: AddNoise below is host-side
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = TakeOrCreateNoise(request, latentShape, seed);
            Tensor latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            noise.Dispose();
            return (latent, sourceLatent);
        }

        Tensor t2iNoise = TakeOrCreateNoise(request, latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, t2iNoise, initSigma);
            t2iNoise.Dispose();
            return (scaled, null);
        }
        return (t2iNoise, null);
    }

    private static Tensor TakeOrCreateNoise(TextToImageRequest request, TensorShape latentShape, int seed)
    {
        if (request.InitialNoise is not null)
        {
            Tensor injected = request.InitialNoise;
            if (!injected.Shape.Equals(latentShape))
                throw new ArgumentException($"InitialNoise shape {injected.Shape} does not match expected latent shape {latentShape}.", nameof(request));
            if (injected.DType != DType.F32)
                throw new ArgumentException($"InitialNoise must be F32; got {injected.DType}.", nameof(request));
            Logs.Info($"SDXL: using injected initial noise tensor (shape={injected.Shape}); seed-based generator skipped.");
            return injected;
        }
        return SeedGenerator.CreateNoise(latentShape, seed);
    }

    /// <summary>Runs the SDXL denoising loop from <paramref name="startStep"/> through <paramref name="totalSteps"/>-1. Handles input-scale, F16 cast, dual-conditioning UNet, CFG, and scheduler step. Returns the final denoised F32 latent. Disposes intermediate latents along the way.
    /// <para>When <paramref name="latentMask"/> is supplied (masked inpaint), after each scheduler step the loop blends in <c>scheduler.AddNoise(sourceLatent, freshNoise, nextStep)</c> on the unmasked region, keeping it on the source's noise trajectory while the masked region is freely denoised. This matches diffusers' <c>StableDiffusionInpaintPipelineLegacy</c> formulation.</para></summary>
    private Tensor RunDenoiseLoop(
        Tensor latent,
        TensorShape latentShape,
        Tensor textEmbeddings,
        Tensor pooledOutput,
        float[] sizeCondition,
        IScheduler scheduler,
        bool useF16,
        int startStep,
        int totalSteps,
        float cfgScale,
        Tensor? sourceLatent,
        Tensor? latentMask,
        int seed,
        IReadOnlyList<ControlNetConditioning>? controlNets,
        RefinerSwapConfig? refiner,
        int swapStep,
        Tensor? clipGForRefiner,
        float[]? refinerSizeConditionPos,
        float[]? refinerSizeConditionNeg,
        IReadOnlyList<IpAdapterConditioning>? ipAdapters,
        Action<GenerationProgress>? onProgress,
        ConditioningSchedule? conditioningSchedule = null)
    {
        bool useStepSwap = refiner is not null && swapStep < totalSteps && clipGForRefiner is not null;
        bool refinerUseF16 = useStepSwap
            && (refiner!.RefinerUnet.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;

        // IP-Adapter setup. v1 honors the first adapter in the list (Swarm UI exposes one
        // slot; the upstream API allows multiple but pipelines don't stack image-attention
        // contributions across them yet). The K/V tensor lists and image tokens are fixed
        // across the whole loop; the per-layer base scale array is built here once from the
        // adapter's weight type, and per-step gating multiplies it to handle the start/end
        // fraction window.
        IpAdapterConditioning? ipa = (ipAdapters is not null && ipAdapters.Count > 0) ? ipAdapters[0] : null;
        Tensor? ipaImageTokens = ipa?.ImageTokens;
        IReadOnlyList<Tensor>? ipaToKIp = null;
        IReadOnlyList<Tensor>? ipaToVIp = null;
        float[]? ipaBaseScalesPerLayer = null;
        if (ipa is not null)
        {
            int layers = ipa.Adapter.CrossAttentionLayerCount;
            Tensor[] kArr = new Tensor[layers];
            Tensor[] vArr = new Tensor[layers];
            for (int li = 0; li < layers; li++)
            {
                kArr[li] = ipa.Adapter.GetToKIpWeight(li);
                vArr[li] = ipa.Adapter.GetToVIpWeight(li);
            }
            ipaToKIp = kArr;
            ipaToVIp = vArr;
            ipaBaseScalesPerLayer = IpAdapterScaleSchedule.Build(ipa.WeightType, ipa.Scale, layers,
                _unet.DownCrossAttentionLayerCount, _unet.MidCrossAttentionLayerCount);
        }
        Logs.Info("Starting SDXL denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < totalSteps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // Refiner phase: from swapStep onward, swap to refiner UNet + CLIP-G-only text
            // emb + refiner ADM (separate cond/uncond aesthetic scores). ControlNet is
            // disabled here — its zero convs are sized for the base UNet's skip count and
            // the per-skip channel widths, so applying them to the refiner's different
            // block schedule would break the residual addition.
            bool inRefinerPhase = useStepSwap && i >= swapStep;
            UNet activeUnet = inRefinerPhase ? refiner!.RefinerUnet : _unet;
            // Per-step conditioning selection (alternation/scheduling) applies only to the base
            // phase; the refiner phase uses its own CLIP-G-only conditioning.
            Tensor baseTextEmb = conditioningSchedule is null
                ? textEmbeddings
                : conditioningSchedule.Variants[conditioningSchedule.Resolve(i, totalSteps)];
            Tensor activeTextEmb = inRefinerPhase ? clipGForRefiner! : baseTextEmb;
            float[] activeSizeCond = inRefinerPhase ? refinerSizeConditionPos! : sizeCondition;
            float[]? activeSizeCondUncond = inRefinerPhase ? refinerSizeConditionNeg : null;
            bool activeUseF16 = inRefinerPhase ? refinerUseF16 : useF16;

            float inputScale = scheduler.ScaleModelInput(i);
            Tensor scaledLatent;
            if (MathF.Abs(inputScale - 1.0f) > 1e-6f)
            {
                scaledLatent = new Tensor(latentShape, DType.F32);
                Backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }

            Tensor unetInput = activeUseF16
                ? DtypeCastHelper.EnsureDtype(Backend, scaledLatent, DType.F16, disposeSourceOnCast: false)
                : scaledLatent;

            // ControlNet (single pass per step, cond branch only — residuals shared across
            // CFG branches when CFG is on. This matches diffusers' guess_mode=True behavior;
            // running CN twice for strict cond/uncond parity is a future optimization).
            // Skip during refiner phase (zero-conv residuals are base-UNet-shaped). Each
            // adapter is additionally gated by its [start, end] step-fraction window.
            Tensor[]? cnDownRes = null;
            Tensor? cnMidRes = null;
            IReadOnlyList<ControlNetConditioning>? activeControlNets = inRefinerPhase
                ? null
                : ControlNetConditioning.FilterActive(controlNets, i, totalSteps);
            if (activeControlNets is not null)
            {
                int seqLenCN = (int)textEmbeddings.Shape[1];
                int hiddenSizeCN = (int)textEmbeddings.Shape[2];
                int pooledDimCN = (int)pooledOutput.Shape[1];
                Tensor condEmbForCN = CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLenCN, hiddenSizeCN);
                Tensor condPooledForCN = CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDimCN);
                (cnDownRes, cnMidRes) = ControlNet.ForwardStacked(Backend, activeControlNets, unetInput, t, condEmbForCN, condPooledForCN, sizeCondition);
                condEmbForCN.Dispose();
                condPooledForCN.Dispose();
            }

            // IP-Adapter: skip during refiner phase (refiner UNet has a different cross-attn
            // shape so the K_ip/V_ip projections wouldn't match). For the base UNet path,
            // gate by the adapter's [start, end] step-fraction window: outside the window,
            // pass null scales so UNet skips the image-attention computation entirely.
            Tensor? activeIpaTokens = null;
            IReadOnlyList<Tensor>? activeIpaK = null;
            IReadOnlyList<Tensor>? activeIpaV = null;
            IReadOnlyList<float>? activeIpaScales = null;
            if (!inRefinerPhase && ipa is not null && ipaBaseScalesPerLayer is not null)
            {
                float gate = IpAdapterScaleSchedule.StepGate(i, totalSteps, ipa.StartFraction, ipa.EndFraction);
                if (gate > 0f)
                {
                    activeIpaTokens = ipaImageTokens;
                    activeIpaK = ipaToKIp;
                    activeIpaV = ipaToVIp;
                    if (gate == 1.0f)
                    {
                        activeIpaScales = ipaBaseScalesPerLayer;
                    }
                    else
                    {
                        // Gate is exclusively 0 or 1 today (boolean window), but keep the
                        // multiply path so a soft-gate / ramp can be added without rewiring.
                        float[] gated = new float[ipaBaseScalesPerLayer.Length];
                        for (int li = 0; li < gated.Length; li++) gated[li] = ipaBaseScalesPerLayer[li] * gate;
                        activeIpaScales = gated;
                    }
                }
            }

            Tensor noisePred;
            if (cfgScale > 1.0f)
            {
                noisePred = ClassifierFreeGuidanceStep(unetInput, t, activeTextEmb, pooledOutput, activeSizeCond, cfgScale, cnDownRes, cnMidRes,
                    overrideUnet: inRefinerPhase ? activeUnet : null,
                    sizeConditionUncond: activeSizeCondUncond,
                    ipaImageTokens: activeIpaTokens, ipaToKIpAll: activeIpaK, ipaToVIpAll: activeIpaV, ipaScalePerLayer: activeIpaScales);
            }
            else
            {
                int seqLen = (int)activeTextEmb.Shape[1];
                int hiddenSize = (int)activeTextEmb.Shape[2];
                Tensor condEmb = CfgHelper.SliceBatchElement(activeTextEmb, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput.Shape[1];
                Tensor condPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = activeUnet.Forward(Backend, unetInput, t, condEmb, condPooled, activeSizeCond, cnDownRes, cnMidRes,
                    activeIpaTokens, activeIpaK, activeIpaV, activeIpaScales);
                condEmb.Dispose();
                condPooled.Dispose();
            }

            if (cnDownRes is not null)
            {
                foreach (Tensor d in cnDownRes) d.Dispose();
                cnMidRes?.Dispose();
            }

            if (unetInput != scaledLatent) unetInput.Dispose();
            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor noisePredF32 = DtypeCastHelper.EnsureF32(Backend, noisePred);

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePredF32, latent, i);
            noisePredF32.Dispose();
            latent.Dispose();
            latent = newLatent;

            // Masked-inpaint blend: keep unmasked region on the source's noise trajectory.
            // newLatent = newLatent * mask + AddNoise(source, fresh_noise, nextStep) * (1 - mask).
            // For the final step, blend with the clean source latent (no further denoising will run).
            if (latentMask is not null && sourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < totalSteps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    noisedSource = new Tensor(latentShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourceLatent, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourceLatent;
                }
                MaskBlendUtilities.BlendChannelsInPlace(latent, noisedSource, latentMask);
                if (Environment.GetEnvironmentVariable("HARTSY_SDXL_DEBUG") == "1")
                {
                    unsafe
                    {
                        float* lp = (float*)latent.DataPointer, np = (float*)noisedSource.DataPointer, mp = (float*)latentMask.DataPointer;
                        int lh = (int)latent.Shape[2], lw = (int)latent.Shape[3];
                        long spatial = (long)lh * lw;
                        double agree = 0, maskMean = 0; long unmasked = 0;
                        for (long p = 0; p < spatial; p++)
                        {
                            maskMean += mp[p];
                            if (mp[p] < 0.01f) { agree += Math.Abs(lp[p] - np[p]); unmasked++; }
                        }
                        Logs.Info($"[SDXLDBG] step {i}: latentMask mean={maskMean / spatial:F4} unmaskedPx={unmasked}/{spatial} " +
                            $"post-blend |latent-noisedSource| on unmasked ch0={(unmasked > 0 ? agree / unmasked : -1):F6}");
                    }
                }
                if (noisedSource != sourceLatent) noisedSource.Dispose();
            }

            stepSw.Stop();
            string cacheInfo = GetBackendCacheStats();
            Logs.Info($"Step {i + 1}/{totalSteps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms{cacheInfo}");
            onProgress?.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.Sdxl,
            });
        }

        return latent;
    }

    /// <summary>Drain-free denoise loop: cond+uncond run as ONE batch-2 UNet forward per step (halving host op-dispatch), the CFG combine + Euler update run in-place on the device-resident latent (<c>IBackend.CfgEulerStep</c>, dt = σ[i+1]−σ[i]), and the step-invariant conditioning (batched text embedding in the UNet dtype, ADM micro-conditioning embedding) is built once before the loop. Zero host round-trips per step — the host CFG/scheduler loops and the per-step conditioning re-slices of the reference path each forced a full GPU pipeline drain. Latent previews are throttled to every 4th step + final (each preview is a deliberate D2H sync). At <c>cfgScale ≤ 1</c> the uncond branch is skipped entirely (batch 1, guidance 1 ⇒ pure cond Euler step). The final latent is host-materialized before returning because the tiled VAE fallback slices it on the host.</summary>
    private Tensor RunDenoiseLoopFused(
        Tensor latent,
        TensorShape latentShape,
        Tensor textEmbeddings,
        Tensor pooledOutput,
        float[] sizeCondition,
        EulerDiscreteScheduler scheduler,
        bool useF16,
        int startStep,
        int totalSteps,
        float cfgScale,
        Action<GenerationProgress>? onProgress)
    {
        bool useCfg = CfgHelper.IsGuidanceActive(cfgScale);
        int batch = useCfg ? 2 : 1;
        int latentC = (int)latentShape[1];
        int latentH = (int)latentShape[2];
        int latentW = (int)latentShape[3];
        DType unetDtype = useF16 ? DType.F16 : DType.F32;

        // Step-invariant conditioning, built once. With CFG the [uncond, cond] batch tensors are used
        // as-is; without it (turbo/distilled checkpoints) the cond element is sliced out once here —
        // the reference loop re-sliced per step, uploading fresh tensors every step.
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];
        int pooledDim = (int)pooledOutput.Shape[1];
        Tensor condText = useCfg ? textEmbeddings : CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor admSource = useCfg ? pooledOutput : CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDim);
        Tensor textForUnet = DtypeCastHelper.EnsureDtype(Backend, condText, unetDtype, disposeSourceOnCast: false);
        bool ownsTextForUnet = !ReferenceEquals(textForUnet, condText);
        Tensor? admEmb = _unet.ComputeAdmEmbedding(Backend, admSource, sizeCondition, batch);

        TensorShape batchedShape = new TensorShape(batch, latentC, latentH, latentW);
        int condRowOffset = latentC * latentH;
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        Logs.Info($"Starting SDXL fused denoising loop (batch={batch}, {unetDtype} activations)...");

        for (int i = startStep; i < totalSteps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            float inputScale = scheduler.ScaleModelInput(i);
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, latent, inputScale);
            Tensor unetInput;
            if (useCfg)
            {
                Tensor batched = new Tensor(batchedShape, DType.F32);
                Backend.Concat(batched, [scaled, scaled], 0);
                scaled.Dispose();
                unetInput = batched;
            }
            else
            {
                unetInput = scaled;
            }
            unetInput = DtypeCastHelper.EnsureDtype(Backend, unetInput, unetDtype);

            Tensor noisePred = _unet.Forward(Backend, unetInput, t, textForUnet, null, default, null, null, null, null, null, null, admEmb);
            unetInput.Dispose();
            Tensor predF32 = DtypeCastHelper.EnsureF32(Backend, noisePred);

            if (useCfg)
            {
                Tensor uncond = new Tensor(latentShape, DType.F32);
                Backend.SliceRows(uncond, predF32, 0);
                Tensor cond = new Tensor(latentShape, DType.F32);
                Backend.SliceRows(cond, predF32, condRowOffset);
                predF32.Dispose();
                Backend.CfgEulerStep(latent, cond, uncond, cfgScale, scheduler.StepDelta(i));
                cond.Dispose();
                uncond.Dispose();
            }
            else
            {
                Backend.CfgEulerStep(latent, predF32, predF32, 1.0f, scheduler.StepDelta(i));
                predF32.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{totalSteps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            bool emitPreview = onProgress is not null && ((i - startStep) % 4 == 3 || i == totalSteps - 1);
            if (emitPreview)
            {
                onProgress!.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.Sdxl,
                });
            }
            else
            {
                onProgress?.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds));
            }
        }

        if (ownsTextForUnet) textForUnet.Dispose();
        if (!ReferenceEquals(condText, textEmbeddings)) condText.Dispose();
        if (!ReferenceEquals(admSource, pooledOutput)) admSource.Dispose();
        admEmb?.Dispose();

        // The in-place CfgEulerStep keeps the latent device-resident; touching DataPointer syncs it
        // back so the tiled-VAE host slicing (and any host consumer) sees the real final state.
        Backend.Sync();
        unsafe { _ = (nint)latent.DataPointer; }
        return latent;
    }

    /// <summary>Compares two token-id arrays for the prompt-embedding cache key.</summary>
    private static bool TokensEqual(int[]? cached, int[] incoming)
        => cached is not null && cached.AsSpan().SequenceEqual(incoming);

    /// <summary>Runs classifier-free guidance for SDXL: noise_pred = uncond + cfg_scale * (cond - uncond). When ControlNet residuals are supplied they're applied to both UNet branches (single CN pass per step, residuals shared — matches diffusers' guess_mode=True; strict per-branch CN passes are a future optimization).
    /// <para>The optional <paramref name="overrideUnet"/> + <paramref name="sizeConditionUncond"/> parameters drive refiner StepSwap mode: during the refiner phase the loop calls this with the refiner UNet and a separate uncond ADM array (so the cond/uncond branches use different aesthetic_score values, matching the refiner's training).</para></summary>
    private Tensor ClassifierFreeGuidanceStep(Tensor latent, float timestep, Tensor textEmbeddings, Tensor pooledOutput, float[] sizeCondition, float cfgScale,
        IReadOnlyList<Tensor>? cnDownRes = null, Tensor? cnMidRes = null,
        UNet? overrideUnet = null, float[]? sizeConditionUncond = null,
        Tensor? ipaImageTokens = null, IReadOnlyList<Tensor>? ipaToKIpAll = null, IReadOnlyList<Tensor>? ipaToVIpAll = null, IReadOnlyList<float>? ipaScalePerLayer = null)
    {
        UNet activeUnet = overrideUnet ?? _unet;
        float[] uncondAdm = sizeConditionUncond ?? sizeCondition;
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];
        int pooledDim = (int)pooledOutput.Shape[1];

        Tensor uncondEmb = CfgHelper.SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor uncondPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 0, pooledDim);
        Tensor condPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDim);

        // Run UNet twice with ADM conditioning (and optional ControlNet residuals + IPA injection).
        // IPA is applied to BOTH branches with the same image tokens — diffusers' IPAdapter
        // attn processor uses the same image-prompt tokens for cond and uncond, so the IPA
        // contribution gets cancelled out by CFG except where Q differs (which is everywhere
        // because Q is from the latent, not the text). Net: IPA influences final output by
        // exactly its scaled image-attention contribution.
        Tensor uncondNoise = activeUnet.Forward(Backend, latent, timestep, uncondEmb, uncondPooled, uncondAdm, cnDownRes, cnMidRes,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        Tensor condNoise = activeUnet.Forward(Backend, latent, timestep, condEmb, condPooled, sizeCondition, cnDownRes, cnMidRes,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        uncondEmb.Dispose();
        condEmb.Dispose();
        uncondPooled.Dispose();
        condPooled.Dispose();

        // UNet output may be F16 — CFG arithmetic runs in F32 (returned to the scheduler).
        Tensor uncondF32 = DtypeCastHelper.EnsureF32(Backend, uncondNoise);
        Tensor condF32 = DtypeCastHelper.EnsureF32(Backend, condNoise);
        Tensor output = CfgHelper.ApplyCfg(uncondF32, condF32, cfgScale);
        uncondF32.Dispose();
        condF32.Dispose();
        return output;
    }

    /// <summary>Gets GPU cache stats string if the backend supports it.</summary>
    private string GetBackendCacheStats()
    {
        System.Reflection.MethodInfo? method = Backend.GetType().GetMethod("GetGpuCacheStats");
        if (method != null)
        {
            (long cachedBytes, long hits, long misses) stats = ((long, long, long))method.Invoke(Backend, null)!;
            return $" (GPU cache: {stats.cachedBytes / 1024 / 1024}MB, hits={stats.hits}, misses={stats.misses})";
        }
        return "";
    }
}
