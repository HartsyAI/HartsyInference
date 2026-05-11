using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Schedulers;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Adapters;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>SDXL text-to-image and image-to-image pipeline. Orchestrates dual CLIP text encoding (CLIP-L + CLIP-G) → UNet denoising with ADM conditioning → VAE decode → RGB image output.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromTokens"/> — same method, single API. Requires a <see cref="VaeEncoder"/> on construction. This also unlocks the cross-model refining pattern: any base pipeline's RGB output can be fed through this pipeline as a refiner via <see cref="ImagePostProcessor.RgbBytesToTensor"/> + img2img with low <c>Strength</c>.</para>
/// </summary>
public sealed unsafe class SdxlPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly float _vaeScalingFactor;
    private int _disposed;

    /// <summary>Creates a new SDXL pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipL">CLIP ViT-L/14 text encoder (text_encoder).</param>
    /// <param name="clipG">OpenCLIP ViT-bigG/14 text encoder (text_encoder_2).</param>
    /// <param name="unet">SDXL UNet (configured with UNetConfig.SdxlBase).</param>
    /// <param name="vaeDecoder">VAE decoder (configured with VaeConfig.Sdxl, scaling factor 0.13025).</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Default: 0.13025 for SDXL.</param>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, float vaeScalingFactor = 0.13025f)
        : this(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder: null, vaeScalingFactor)
    {
    }

    /// <summary>Creates a new SDXL pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, float vaeScalingFactor = 0.13025f)
    {
        _backend = backend;
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
        IReadOnlyList<IpAdapterConditioning>? ipAdapters = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        // Img2img: validate source shape + handle strength=0 short-circuit BEFORE any model work.
        int startStep = 0;
        Tensor? maskPixel = null;
        if (request is ImageToImageRequest img2img)
        {
            Tensor src = img2img.SourceImage;
            if (src.Shape.Rank != 4 || src.Shape[0] != 1 || src.Shape[1] != 3 ||
                src.Shape[2] != height || src.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {src.Shape}.",
                    nameof(request));
            }

            if (img2img.Mask is not null)
            {
                Tensor mk = img2img.Mask;
                if (mk.Shape.Rank != 4 || mk.Shape[0] != 1 || mk.Shape[1] != 1 ||
                    mk.Shape[2] != height || mk.Shape[3] != width)
                {
                    throw new ArgumentException(
                        $"Mask shape must be [1, 1, {height}, {width}] (matching source); got {mk.Shape}.",
                        nameof(request));
                }
                if (mk.DType != DType.F32)
                {
                    throw new ArgumentException($"Mask must be F32 in [0, 1]; got {mk.DType}.", nameof(request));
                }
                maskPixel = mk;
            }

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), width, height, seed);
            }
        }

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
                refinerSizeConditionPos = [request.Height, request.Width, 0f, 0f, refiner.AestheticScore];
                refinerSizeConditionNeg = [request.Height, request.Width, 0f, 0f, refiner.NegativeAestheticScore];
            }
        }

        string mode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                    : isImg2Img ? $"img2img (start={startStep}/{steps})"
                    : "txt2img";
        if (useStepSwap) mode += $" + refiner@{swapStep}/{steps}";
        Logs.Info($"SDXL {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Dual CLIP text encoding (shared by both paths)
        Logs.Info("Encoding text with dual CLIP encoders...");
        int[][] batchTokenIdsL = [negativePromptTokenIdsL, promptTokenIdsL];
        (Tensor clipLHidden, _) = _clipL.EncodePenultimate(_backend, batchTokenIdsL, [0, 0]);

        int[][] batchTokenIdsG = [negativePromptTokenIdsG, promptTokenIdsG];
        int[] eosPositions = [negativeEosPositionG, promptEosPositionG];
        (Tensor clipGHidden, Tensor? pooledOutput) = _clipG.EncodePenultimate(_backend, batchTokenIdsG, eosPositions);

        Tensor textEmbeddings = ConcatAlongLastDim(clipLHidden, clipGHidden);
        clipLHidden.Dispose();
        // Keep clipGHidden alive when StepSwap is active — refiner UNet uses CLIP-G alone
        // (CrossAttentionDim=1280) instead of the concat (2048). Disposed at end of pipeline.
        Tensor? clipGForRefiner = useStepSwap ? clipGHidden : null;
        if (!useStepSwap) clipGHidden.Dispose();
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. ADM size conditioning (orig_size = target_size = request resolution; no crop)
        float[] sizeCondition =
        [
            request.Height, request.Width,
            0f, 0f,
            request.Height, request.Width,
        ];

        // 3. Set up scheduler
        IScheduler scheduler = CreateScheduler(request.Scheduler);
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
        bool useF16 = (_unet.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;
        latent = RunDenoiseLoop(latent, latentShape, textEmbeddings, pooledOutput!, sizeCondition, scheduler, useF16, startStep, totalSteps: steps, cfgScale, sourceLatent, latentMask, seed, controlNets,
            refiner, swapStep, clipGForRefiner, refinerSizeConditionPos, refinerSizeConditionNeg, ipAdapters, onProgress);

        textEmbeddings.Dispose();
        pooledOutput?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();
        clipGForRefiner?.Dispose();

        // 6. VAE decode â free UNet weights to reclaim VRAM for high-res VAE conv2d buffers.
        // CLIP-L/CLIP-G stay resident (they don't expose EnumerateWeights yet); BF16 VAE
        // is the same byte count as the previously-broken F16 path so no extra pressure.
        _backend.Sync();
        _backend.FreeWeights(_unet.EnumerateWeights());

        // Match the latent dtype to the loaded VAE weight dtype. SDXL VAE F16 is broken
        // (resnet activations overflow â NaN â black output); the loader now casts SDXL
        // VAE weights to BF16 (Ampere+) or F32 (older HW) per the ComfyUI policy. Cast the
        // latent to match â otherwise the kernel dispatch reads weights of one dtype as
        // another and produces garbage. See PHASE_3_DEVIATIONS.md ("F16 has been observed
        // to produce all-black output").
        DType vaeDtype = _vaeDecoder.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32;
        Logs.Verbose($"Decoding latents to image (VAE dtype={vaeDtype})...");
        Stopwatch vaeSw = Stopwatch.StartNew();

        Tensor vaeInput = latent;
        if (latent.DType != vaeDtype)
        {
            vaeInput = new Tensor(latent.Shape, vaeDtype);
            if (vaeDtype == DType.F16) _backend.CastToF16(vaeInput, latent);
            else if (vaeDtype == DType.BF16) _backend.CastToBf16(vaeInput, latent);
            else if (vaeDtype == DType.F32) _backend.CastToF32(vaeInput, latent);
            else throw new InvalidOperationException($"Unsupported VAE dtype: {vaeDtype}");
            latent.Dispose();
        }

        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Tensor image = _vaeDecoder.DecodeTiled(_backend, vaeInput);
        vaeInput.Dispose();
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
            Tensor sourceLatent = _vaeEncoder!.Encode(_backend, img2img.SourceImage);
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
            _backend.Scale(scaled, t2iNoise, initSigma);
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
        Action<GenerationProgress>? onProgress)
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
            ipaBaseScalesPerLayer = IpAdapterScaleSchedule.Build(ipa.WeightType, ipa.Scale, layers);
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
            Tensor activeTextEmb = inRefinerPhase ? clipGForRefiner! : textEmbeddings;
            float[] activeSizeCond = inRefinerPhase ? refinerSizeConditionPos! : sizeCondition;
            float[]? activeSizeCondUncond = inRefinerPhase ? refinerSizeConditionNeg : null;
            bool activeUseF16 = inRefinerPhase ? refinerUseF16 : useF16;

            float inputScale = scheduler.ScaleModelInput(i);
            Tensor scaledLatent;
            if (MathF.Abs(inputScale - 1.0f) > 1e-6f)
            {
                scaledLatent = new Tensor(latentShape, DType.F32);
                _backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }

            Tensor unetInput = scaledLatent;
            if (activeUseF16 && scaledLatent.DType != DType.F16)
            {
                unetInput = new Tensor(scaledLatent.Shape, DType.F16);
                _backend.CastToF16(unetInput, scaledLatent);
            }

            // ControlNet (single pass per step, cond branch only — residuals shared across
            // CFG branches when CFG is on. This matches diffusers' guess_mode=True behavior;
            // running CN twice for strict cond/uncond parity is a future optimization).
            // Skip during refiner phase (zero-conv residuals are base-UNet-shaped).
            Tensor[]? cnDownRes = null;
            Tensor? cnMidRes = null;
            if (!inRefinerPhase && controlNets is not null && controlNets.Count > 0)
            {
                int seqLenCN = (int)textEmbeddings.Shape[1];
                int hiddenSizeCN = (int)textEmbeddings.Shape[2];
                int pooledDimCN = (int)pooledOutput.Shape[1];
                Tensor condEmbForCN = SliceBatchElement(textEmbeddings, 1, seqLenCN, hiddenSizeCN);
                Tensor condPooledForCN = SliceBatchElement1D(pooledOutput, 1, pooledDimCN);
                (cnDownRes, cnMidRes) = RunControlNets(controlNets, unetInput, t, condEmbForCN, condPooledForCN, sizeCondition);
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
                Tensor condEmb = SliceBatchElement(activeTextEmb, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput.Shape[1];
                Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = activeUnet.Forward(_backend, unetInput, t, condEmb, condPooled, activeSizeCond, cnDownRes, cnMidRes,
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

            Tensor noisePredF32 = noisePred;
            if (noisePred.DType == DType.F16)
            {
                noisePredF32 = new Tensor(noisePred.Shape, DType.F32);
                _backend.CastToF32(noisePredF32, noisePred);
                noisePred.Dispose();
            }

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

        // Split text embeddings: [0] = negative (uncond), [1] = positive (cond)
        Tensor uncondEmb = SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor uncondPooled = SliceBatchElement1D(pooledOutput, 0, pooledDim);
        Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);

        // Run UNet twice with ADM conditioning (and optional ControlNet residuals + IPA injection).
        // IPA is applied to BOTH branches with the same image tokens — diffusers' IPAdapter
        // attn processor uses the same image-prompt tokens for cond and uncond, so the IPA
        // contribution gets cancelled out by CFG except where Q differs (which is everywhere
        // because Q is from the latent, not the text). Net: IPA influences final output by
        // exactly its scaled image-attention contribution.
        Tensor uncondNoise = activeUnet.Forward(_backend, latent, timestep, uncondEmb, uncondPooled, uncondAdm, cnDownRes, cnMidRes,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        Tensor condNoise = activeUnet.Forward(_backend, latent, timestep, condEmb, condPooled, sizeCondition, cnDownRes, cnMidRes,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        uncondEmb.Dispose();
        condEmb.Dispose();
        uncondPooled.Dispose();
        condPooled.Dispose();

        // CFG: output = uncond + scale * (cond - uncond)
        // UNet output may be F16 — cast to F32 for CPU-side arithmetic, return F32 for scheduler
        Tensor uncondF32 = uncondNoise;
        Tensor condF32 = condNoise;
        if (uncondNoise.DType == DType.F16)
        {
            uncondF32 = new Tensor(uncondNoise.Shape, DType.F32);
            _backend.CastToF32(uncondF32, uncondNoise);
            uncondNoise.Dispose();
            condF32 = new Tensor(condNoise.Shape, DType.F32);
            _backend.CastToF32(condF32, condNoise);
            condNoise.Dispose();
        }

        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* uncPtr = (float*)uncondF32.DataPointer;
        float* conPtr = (float*)condF32.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)latent.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);
        }

        uncondF32.Dispose();
        condF32.Dispose();

        return output;
    }

    /// <summary>Runs every supplied ControlNet at the current step and sums their residuals into a single set. Each ControlNet sees the same latent + timestep + cond text embedding and produces its own per-skip residual stack and mid residual; this helper collapses them by element-wise addition so the UNet path doesn't need to know how many ControlNets are stacked.</summary>
    private (Tensor[] downResiduals, Tensor midResidual) RunControlNets(
        IReadOnlyList<ControlNetConditioning> controlNets,
        Tensor latent,
        float timestep,
        Tensor condTextEmb,
        Tensor condPooled,
        ReadOnlySpan<float> sizeCondition)
    {
        ControlNetConditioning first = controlNets[0];
        (Tensor[] down, Tensor mid) = first.Adapter.Forward(_backend, latent, first.ConditionImage, timestep, condTextEmb, condPooled, sizeCondition, first.Scale);
        for (int c = 1; c < controlNets.Count; c++)
        {
            ControlNetConditioning next = controlNets[c];
            (Tensor[] downNext, Tensor midNext) = next.Adapter.Forward(_backend, latent, next.ConditionImage, timestep, condTextEmb, condPooled, sizeCondition, next.Scale);
            if (downNext.Length != down.Length)
            {
                foreach (Tensor d in downNext) d.Dispose();
                midNext.Dispose();
                throw new InvalidOperationException(
                    $"Stacked ControlNet residual count mismatch: {down.Length} vs {downNext.Length}. " +
                    "All stacked ControlNets must target the same base UNet config.");
            }
            for (int i = 0; i < down.Length; i++)
            {
                Tensor sum = new Tensor(down[i].Shape, down[i].DType);
                _backend.Add(sum, down[i], downNext[i]);
                down[i].Dispose();
                downNext[i].Dispose();
                down[i] = sum;
            }
            Tensor sumMid = new Tensor(mid.Shape, mid.DType);
            _backend.Add(sumMid, mid, midNext);
            mid.Dispose();
            midNext.Dispose();
            mid = sumMid;
        }
        return (down, mid);
    }

    /// <summary>Concatenates two [B, seqLen, D1] and [B, seqLen, D2] tensors along the last dimension → [B, seqLen, D1+D2].</summary>
    private static Tensor ConcatAlongLastDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, seqLen, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;

                for (int d = 0; d < dimA; d++)
                {
                    outPtr[outOffset + d] = aPtr[aOffset + d];
                }
                for (int d = 0; d < dimB; d++)
                {
                    outPtr[outOffset + dimA + d] = bPtr[bOffset + d];
                }
            }
        }

        return output;
    }

    /// <summary>Extracts a single element from the batch dimension of a [B, seqLen, hiddenSize] tensor.</summary>
    private static Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;

        for (int i = 0; i < elements; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }

        return slice;
    }

    /// <summary>Extracts a single element from the batch dimension of a [B, dim] tensor → [1, dim].</summary>
    private static Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;

        for (int i = 0; i < dim; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }

        return slice;
    }

    /// <summary>Creates the requested scheduler.</summary>
    private static IScheduler CreateScheduler(string? name)
    {
        return (name?.ToLowerInvariant()) switch
        {
            "ddim" => new DdimScheduler(),
            "dpm++2m" or "dpmpp2m" => new DpmPlusPlus2MScheduler(),
            _ => new EulerDiscreteScheduler(),
        };
    }

    /// <summary>Gets GPU cache stats string if the backend supports it.</summary>
    private string GetBackendCacheStats()
    {
        System.Reflection.MethodInfo? method = _backend.GetType().GetMethod("GetGpuCacheStats");
        if (method != null)
        {
            (long cachedBytes, long hits, long misses) stats = ((long, long, long))method.Invoke(_backend, null)!;
            return $" (GPU cache: {stats.cachedBytes / 1024 / 1024}MB, hits={stats.hits}, misses={stats.misses})";
        }
        return "";
    }

    /// <summary>Evicts GPU weight cache if the backend supports it (CudaBackend).</summary>
    private void EvictBackendCache(string stage)
    {
        System.Reflection.MethodInfo? method = _backend.GetType().GetMethod("EvictGpuCache");
        method?.Invoke(_backend, null);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components (shared resources).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
