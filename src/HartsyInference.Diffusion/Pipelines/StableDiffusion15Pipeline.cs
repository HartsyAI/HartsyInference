using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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

/// <summary>Stable Diffusion 1.5 text-to-image and image-to-image pipeline. Orchestrates CLIP text encoder → UNet denoising loop → VAE decode → RGB image output. Provide a <see cref="VaeEncoder"/> on construction to enable img2img.</summary>
public sealed class StableDiffusion15Pipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _textEncoder;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;

    /// <summary>Creates a new SD1.5 pipeline. Img2img is unavailable (will throw); use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public StableDiffusion15Pipeline(IBackend backend, ClipTextEncoder textEncoder, UNet unet, VaeDecoder vaeDecoder)
        : this(backend, textEncoder, unet, vaeDecoder, vaeEncoder: null)
    {
    }

    /// <summary>Creates a new SD1.5 pipeline with both VAE halves loaded. Required for img2img.</summary>
    public StableDiffusion15Pipeline(IBackend backend, ClipTextEncoder textEncoder, UNet unet, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
    }

    /// <summary>Generates an image from pre-tokenized input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image. Initial latent is fresh Gaussian noise scaled by the scheduler's <c>InitialNoiseSigma</c>; denoise from step 0.</item>
    /// <item><see cref="ImageToImageRequest"/> → image-to-image. The source image is encoded via the VAE encoder, fresh noise is injected at <c>sigma[startStep]</c> via <c>scheduler.AddNoise</c>, and denoising runs from <c>startStep = steps - round(steps * Strength)</c>. Requires a pipeline constructed with a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// The two paths only differ in the initial latent and the start step — text encoding, denoise loop, VAE decode, and RGB conversion are identical. Strength=0 short-circuits to a byte-identical pass-through.
    /// </summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativePromptTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        IReadOnlyList<IpAdapterConditioning>? ipAdapters = null,
        ConditioningSchedule? conditioningSchedule = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Sd15.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }

        string mode = isImg2Img ? $"img2img (start={plan.StartStep}/{steps})" : "txt2img";
        Logs.Info($"SD1.5 {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Encode text
        Logs.Info("Encoding text prompt...");
        int[][] batchTokenIds = [negativePromptTokenIds, promptTokenIds];
        Tensor textEmbeddings = _textEncoder.Encode(Backend, batchTokenIds, request.ClipSkip ?? 1);
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. Set up scheduler (needed for both paths' initial-latent prep)
        IScheduler scheduler = SchedulerFactory.Create(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 3. Build initial latent — t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at startStep.
        Tensor latent = BuildInitialLatent(request, scheduler, latentShape, seed, plan.StartStep);

        // 4. Denoise loop (both paths run the same loop from their respective startStep)
        latent = RunDenoiseLoop(latent, latentShape, textEmbeddings, scheduler, plan.StartStep, totalSteps: steps, cfgScale, ipAdapters, onProgress, conditioningSchedule);

        textEmbeddings.Dispose();

        // 5. VAE decode (tiled — caps im2col workspace at ~2.4 GB per tile)
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 6. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SD1.5 {mode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the starting latent for the denoise loop. For text-to-image this is fresh Gaussian noise scaled by the scheduler's initial sigma. For image-to-image this is the VAE-encoded source plus fresh noise injected at sigma[startStep] via <c>scheduler.AddNoise</c>.</summary>
    private Tensor BuildInitialLatent(TextToImageRequest request, IScheduler scheduler, TensorShape latentShape, int seed, int startStep)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = TakeOrCreateNoise(request, latentShape, seed);
            Tensor latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            sourceLatent.Dispose();
            noise.Dispose();
            return latent;
        }

        // Text-to-image: scale fresh noise by the scheduler's initial sigma.
        Tensor t2iNoise = TakeOrCreateNoise(request, latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, t2iNoise, initSigma);
            t2iNoise.Dispose();
            return scaled;
        }
        return t2iNoise;
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
            Logs.Info($"SD1.5: using injected initial noise tensor (shape={injected.Shape}); seed-based generator skipped.");
            return injected;
        }
        return SeedGenerator.CreateNoise(latentShape, seed);
    }

    /// <summary>Runs the diffusion denoising loop. Iterates <c>i</c> from <paramref name="startStep"/> through <paramref name="totalSteps"/>-1, applying scheduler input scaling, the UNet (with optional CFG), and one scheduler step per iteration. Returns the final denoised latent. Disposes intermediate latents along the way.</summary>
    private Tensor RunDenoiseLoop(
        Tensor latent,
        TensorShape latentShape,
        Tensor textEmbeddings,
        IScheduler scheduler,
        int startStep,
        int totalSteps,
        float cfgScale,
        IReadOnlyList<IpAdapterConditioning>? ipAdapters,
        Action<GenerationProgress>? onProgress,
        ConditioningSchedule? conditioningSchedule = null)
    {
        // IP-Adapter setup (same shape as SDXL — single adapter honored, weight-type +
        // start/end window applied per step).
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

        Logs.Info("Starting denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < totalSteps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // Scale model input (Euler: 1/sqrt(sigma^2+1), others: 1.0)
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

            // Per-step IPA gating
            Tensor? activeIpaTokens = null;
            IReadOnlyList<Tensor>? activeIpaK = null;
            IReadOnlyList<Tensor>? activeIpaV = null;
            IReadOnlyList<float>? activeIpaScales = null;
            if (ipa is not null && ipaBaseScalesPerLayer is not null)
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
                        float[] gated = new float[ipaBaseScalesPerLayer.Length];
                        for (int li = 0; li < gated.Length; li++) gated[li] = ipaBaseScalesPerLayer[li] * gate;
                        activeIpaScales = gated;
                    }
                }
            }

            // Per-step conditioning selection for alternation [a|b] / scheduling [a:b:when].
            Tensor stepEmbeddings = conditioningSchedule is null
                ? textEmbeddings
                : conditioningSchedule.Variants[conditioningSchedule.Resolve(i, totalSteps)];

            Tensor noisePred;
            if (cfgScale > 1.0f)
            {
                noisePred = ClassifierFreeGuidanceStep(scaledLatent, t, stepEmbeddings, cfgScale,
                    activeIpaTokens, activeIpaK, activeIpaV, activeIpaScales);
            }
            else
            {
                int seqLen = (int)stepEmbeddings.Shape[1];
                int hiddenSize = (int)stepEmbeddings.Shape[2];
                Tensor condEmb = CfgHelper.SliceBatchElement(stepEmbeddings, 1, seqLen, hiddenSize);
                noisePred = _unet.Forward(Backend, scaledLatent, t, condEmb, null, default,
                    null, null,
                    activeIpaTokens, activeIpaK, activeIpaV, activeIpaScales);
                condEmb.Dispose();
            }

            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{totalSteps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.Sd15,
            });
        }

        return latent;
    }

    /// <summary>Runs classifier-free guidance: noise_pred = uncond + cfg_scale * (cond - uncond). Optionally injects IP-Adapter image-attention contribution into both UNet branches.</summary>
    private Tensor ClassifierFreeGuidanceStep(Tensor latent, float timestep, Tensor textEmbeddings, float cfgScale,
        Tensor? ipaImageTokens = null, IReadOnlyList<Tensor>? ipaToKIpAll = null, IReadOnlyList<Tensor>? ipaToVIpAll = null, IReadOnlyList<float>? ipaScalePerLayer = null)
    {
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        Tensor uncondEmb = CfgHelper.SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);

        // Run UNet twice. SD1.5 has no SDXL ADM conditioning; pooled / sizeCondition / CN
        // residuals stay null. IPA params are passed identically to both branches.
        Tensor uncondNoise = _unet.Forward(Backend, latent, timestep, uncondEmb, null, default,
            null, null,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        Tensor condNoise = _unet.Forward(Backend, latent, timestep, condEmb, null, default,
            null, null,
            ipaImageTokens, ipaToKIpAll, ipaToVIpAll, ipaScalePerLayer);
        uncondEmb.Dispose();
        condEmb.Dispose();

        Tensor output = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
        uncondNoise.Dispose();
        condNoise.Dispose();
        return output;
    }
}
