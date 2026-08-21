using HartsyInference.Diffusion.Sampling;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Lumina-Image-2.0 text-to-image and image-to-image pipeline (Alpha-VLLM, Apache 2.0). Accepts pre-computed Gemma-2 caption embeddings (text-encoder forward is owned by a separate component, since HartsyInference's <see cref="Models.TextEncoders.LlamaStyleEncoder"/> does not yet implement Gemma-2-specific features like GeGLU MLP and attention soft-capping) and orchestrates the NextDiT transformer with the checkpoint's static-shift flow-match Euler scheduler.
/// <para>The diffusers pipeline (<c>pipeline_lumina2.py</c>) inverts the timestep before feeding it to the transformer (<c>1 - t / num_train_timesteps</c>) and negates the predicted velocity before the scheduler step — both behaviors are replicated here for parity. Default sampling: 30 steps, cfg_scale=4.0 with a negative prompt (matches the diffusers default).</para>
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromEmbeddings"/>. Requires a <see cref="VaeEncoder"/> (16-channel Flux VAE, <c>VaeConfig.Flux</c>) on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step latent blend + final pixel recomposite, same pattern as <see cref="ZImagePipeline"/>).</para>
/// </summary>
public sealed unsafe class Lumina2Pipeline : DiffusionPipelineBase
{
    private readonly Lumina2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Lumina2Config _config;

    /// <summary>Creates a Lumina-Image-2.0 pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public Lumina2Pipeline(IBackend backend, Lumina2Transformer transformer, VaeDecoder vaeDecoder, Lumina2Config config)
        : this(backend, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Lumina-Image-2.0 pipeline with both VAE halves loaded. Required for img2img / inpaint.</summary>
    public Lumina2Pipeline(IBackend backend, Lumina2Transformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, Lumina2Config config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Gemma-2 caption embeddings. CFG is applied when <paramref name="cfgScale"/> &gt; 1.0 and a negative-prompt embedding is provided.</summary>
    /// <param name="captionEmbeddings">Last hidden state of Gemma-2-2B applied to the prompt: [B, capLen, 2304]. The Lumina 2.0 system prompt prefix should already be applied upstream.</param>
    /// <param name="request">Generation parameters (Width, Height, Steps, Seed).</param>
    /// <param name="cfgScale">Classifier-free guidance scale. Lumina 2.0's diffusers default is 4.0 with a negative prompt.</param>
    /// <param name="negativeCaptionEmbeddings">Negative-prompt embeddings for CFG. Required when <paramref name="cfgScale"/> &gt; 1.0.</param>
    /// <param name="onProgress">Optional progress callback per step.</param>
    /// <param name="cfgNormalization">Lumina-2.0's <c>cfg_normalization</c> (diffusers default TRUE): rescale the guided prediction back to the conditional's per-token L2 norm. See <see cref="CfgHelper.ApplyCfgNormalized"/>.</param>
    /// <param name="cfgTruncRatio">Lumina-2.0's <c>cfg_trunc_ratio</c> (diffusers default 1.0): for steps where the step-progress fraction <c>(i+1)/steps</c> exceeds this ratio, guidance is skipped and only the conditional prediction is used. At 1.0 the gate never fires (never-skip), so it is a no-op at the default.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 4.0f,
        Tensor? negativeCaptionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null,
        bool cfgNormalization = true,
        float cfgTruncRatio = 1.0f)
    {
        // Sampler selection is NOT wired on this family (2026-08-20 audit). Two things block it, and the second is
        // decisive: Lumina-Image 2.0 is NextDiT so it predicts -v (handled now by PredictionType.NegatedFlowVelocity),
        // but its PRODUCTION default runs Backend.CfgNormalizedEulerStep — a fused normalized-CFG + Euler op with no
        // pair-form decomposition on the backend. Routing that through the sampler would silently DROP the CFG
        // normalization, which is changed output rather than changed cost. `cfgTruncRatio` also makes fused-vs-plain
        // a per-step decision, so a partial conversion would interleave direct and sampler steps and corrupt a
        // stateful sampler's history. It converts once /Sampling/ grows a normalized combine.
        if (FlowMatchSampling.IsNonDefault(request.Scheduler))
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' is not available on Lumina-Image 2.0 — its CFG normalization "
                + "is fused into the device Euler step, which the sampler seam cannot express. Leave the sampler unset.");
        }

        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0. Lumina 2.0's diffusers pipeline always uses a negative prompt at cfg=4.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Lumina2.Width;
        int height = request.Height ?? GenerationDefaults.Lumina2.Height;
        int spatialMultiple = checked(_config.VaeDownscaleFactor * _config.PatchSize);
        if (width <= 0 || height <= 0 || width % spatialMultiple != 0 || height % spatialMultiple != 0)
            throw new ArgumentException(
                $"Lumina 2.0 width and height must be positive multiples of {spatialMultiple} " +
                $"(VAE downscale {_config.VaeDownscaleFactor} × patch size {_config.PatchSize}); got {width}x{height}.",
                nameof(request));
        int latentH = height / _config.VaeDownscaleFactor;
        int latentW = width / _config.VaeDownscaleFactor;
        int steps = request.Steps ?? GenerationDefaults.Lumina2.Steps;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string opMode = isMaskedInpaint ? $"inpaint (start={plan.StartStep}/{steps})"
                      : isImg2Img ? $"img2img (start={plan.StartStep}/{steps})"
                      : "t2i";
        Logs.Info($"Lumina 2.0 {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);

        // Alpha-VLLM/Lumina-Image-2.0's scheduler contract is static shift=6 with dynamic shifting disabled.
        // Flux-style image-sequence-dependent shifting materially changes Lumina's denoise trajectory.
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        (Tensor latent, Tensor? sourceLatent) = BuildInitialLatent(request, scheduler, latentShape, seed, plan.StartStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // DiT sharding: asymmetric preload — Backend gets the shared weights (embedders + BOTH refiner stacks)
        // PLUS its main-layer block range, DitShardBackend gets ONLY its block range. The unsharded path relies
        // entirely on per-op auto-promotion (as before this change) — this bulk upload only fires when sharding
        // is active, avoiding a per-op cache-miss H2D transfer on the first denoise step of the range split.
        if (DitShardBackend is not null)
        {
            Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
        }

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = plan.StartStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Lumina 2.0's diffusers pipeline inverts the timestep before feeding it to the transformer
            // (`current_timestep = 1 - t/num_train_timesteps`). See pipeline_lumina2.py:757.
            float invertedSigma = 1.0f - sigma;
            Tensor velocity = RunForward(latent, captionEmbeddings, invertedSigma);

            // VALIDATION-PENDING: verify against diffusers Lumina2Pipeline (pipeline_lumina2.py:724-757):
            // do_classifier_free_truncation = (i+1)/num_inference_steps > cfg_trunc_ratio. When truncated, guidance is
            // skipped and the conditional prediction is used directly (noise_pred = noise_pred_cond). Otherwise CFG is
            // combined and, when cfg_normalization is on, rescaled back to the conditional's per-token L2 norm.
            bool doClassifierFreeTruncation = (float)(i + 1) / steps > cfgTruncRatio;
            Tensor? uncondVelocity = null;
            try
            {
                // Lumina negates the model velocity before FlowMatch Euler. Folding that negation into
                // delta=-dt keeps the carried latent and both predictions device-resident. The normalized
                // path reduces independently over the last latent dimension, exactly like diffusers.
                float delta = -scheduler.Dt(i);
                if (cfgScale > 1.0f && !doClassifierFreeTruncation)
                {
                    uncondVelocity = RunForward(latent, negativeCaptionEmbeddings!, invertedSigma);
                    if (cfgNormalization)
                        Backend.CfgNormalizedEulerStep(latent, velocity, uncondVelocity, cfgScale, delta);
                    else
                        Backend.CfgEulerStep(latent, velocity, uncondVelocity, cfgScale, delta);
                }
                else
                {
                    Backend.CfgEulerStep(latent, velocity, velocity, 1.0f, delta);
                }
            }
            finally
            {
                uncondVelocity?.Dispose();
                velocity.Dispose();
            }

            // Masked-inpaint blend: keep unmasked region on the source's flow-matching trajectory
            // by re-noising the source latent at the next step's sigma. Final step blends with the
            // clean source — no further denoising follows.
            if (latentMask is not null && sourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
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
            Logs.Debug($"Lumina 2.0 step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        sourceLatent?.Dispose();
        latentMask?.Dispose();

        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Lumina 2.0 {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma (Lumina 2.0's flow-match scheduler at sigma_init typically yields ~1.0 so this is a near-identity scaling). Img2img: VaeEncoder.Encode(source) combined with fresh noise via flow-matching AddNoise at sigma[startStep].
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean source latent is returned alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request, FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep, bool keepSourceLatent)
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
            noise.Dispose();
            if (keepSourceLatent)
            {
                return (latent, sourceLatent);
            }
            sourceLatent.Dispose();
            return (latent, null);
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

    /// <summary>Routes one denoise step through <see cref="DitShardBackend"/>'s block-range split when
    /// configured, else the normal single-backend path.</summary>
    private Tensor RunForward(Tensor latent, Tensor captionEmbeddings, float invertedSigma)
    {
        if (DitShardBackend is not null)
        {
            return _transformer.ForwardSharded(Backend, DitShardBackend, latent, captionEmbeddings, invertedSigma, DitShardSplitBlock);
        }
        return _transformer.Forward(Backend, latent, captionEmbeddings, invertedSigma);
    }

}
