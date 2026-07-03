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

/// <summary>AuraFlow text-to-image pipeline (<c>AuraFlowPipeline</c>). Encodes prompts with Pile-T5-XL (joint_attention_dim=2048), runs the AuraFlow MMDiT transformer with classifier-free guidance, then decodes with the SDXL-compatible 4-channel VAE. Uses a static-shift FlowMatch scheduler. Mirrors the diffusers pipeline at <c>diffusers/pipelines/aura_flow/pipeline_aura_flow.py</c>.</summary>
public sealed class AuraFlowPipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly AuraFlowTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly AuraFlowConfig _config;
    private readonly float _schedulerShift;

    /// <summary>Creates a new AuraFlow pipeline with all components pre-loaded. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">Pile-T5-XL text encoder (output dim 2048, max length 256).</param>
    /// <param name="transformer">AuraFlow MMDiT transformer.</param>
    /// <param name="vaeDecoder">SDXL VAE decoder (<see cref="VaeConfig.Sdxl"/>; 4-channel latent, scaling=0.13025).</param>
    /// <param name="config">AuraFlow configuration (use <see cref="AuraFlowConfig.V03"/> for fal/AuraFlow-v0.3).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. AuraFlow uses a static shift; default 1.73 matches the diffusers config.</param>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, AuraFlowConfig config, float schedulerShift = 1.73f)
        : this(backend, t5, transformer, vaeDecoder, vaeEncoder: null, config, schedulerShift)
    {
    }

    /// <summary>Creates a new AuraFlow pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Configure the encoder with <see cref="VaeConfig.AuraFlow"/> so its output is already in the transformer's latent space.</summary>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, AuraFlowConfig config, float schedulerShift = 1.73f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized Pile-T5-XL input with optional CFG. An <see cref="ImageToImageRequest"/> selects img2img: the source is VAE-encoded and noised via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a <see cref="VaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step latent blend + final pixel recomposite). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from Pile-T5-XL tokenizer (max length 256).</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs (same length as <paramref name="promptTokenIdsT5"/>).</param>
    /// <param name="promptAttentionMaskT5">Optional T5 attention mask for the prompt (1=attend, 0=pad).</param>
    /// <param name="negativeAttentionMaskT5">Optional T5 attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.AuraFlow.Resolve(request);
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

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "txt2img";
        Logs.Info($"AuraFlow {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with Pile-T5-XL ───────────────────────────────
        Logs.Info("Encoding text with Pile-T5-XL...");

        // Bulk-upload T5 weights once so the encoder's many kernel launches don't each pay a
        // per-op cache-miss H2D transfer. Backends without a weight cache no-op. Paired with
        // the FreeWeights below the VAE handoff so VRAM cycles cleanly.
        Backend.PreloadWeights(_t5.EnumerateWeights());

        bool useCfg = cfgScale > 1.0f;

        int[][] batchT5 = [promptTokenIdsT5];
        int[][]? batchMask = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor condContext = _t5.Encode(Backend, batchT5, batchMask);

        Tensor? uncondContext = null;
        if (useCfg)
        {
            int[][] negBatchT5 = [negativePromptTokenIdsT5];
            int[][]? negBatchMask = negativeAttentionMaskT5 is not null ? [negativeAttentionMaskT5] : null;
            uncondContext = _t5.Encode(Backend, negBatchT5, negBatchMask);
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Set up flow-match scheduler (static shift) ────────────────
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial latent [1, 4, latentH, latentW] (t2i: noise * initSigma; img2img: encode + AddNoise at sigma[startStep]) ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        (Tensor latent, Tensor? sourceLatent) =
            BuildInitialLatent(request, scheduler, latentShape, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // ── 4. Denoising loop ────────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop (touched every step, every
        // block — without this the first step would pay cache-miss overhead per parameter).
        // Paired with FreeWeights at the VAE handoff so VRAM cycles between generations.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting AuraFlow denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor uncondNoise = _transformer.Forward(Backend, latent, t, uncondContext!);
                Tensor condNoise = _transformer.Forward(Backend, latent, t, condContext);
                noisePred = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
                uncondNoise.Dispose();
                condNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condContext);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
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
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.AuraFlow,
            });
        }

        condContext.Dispose();
        uncondContext?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        AuraFlowTransformer.DumpFinalLatent(latent);

        // Free transformer + T5 weights from GPU before VAE decode (mirrors SD3 pattern;
        // the SDXL VAE im2col buffers are large and the transformer is huge — we OOM on
        // a 12 GB card without this eviction).
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        Backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. VAE decode ────────────────────────────────────────────────
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
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
        Logs.Info($"AuraFlow {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: VaeEncoder.Encode(source) (SDXL-family
    /// 4-channel VAE, scaling applied inside the encoder) combined with fresh noise via flow-matching AddNoise at
    /// sigma[startStep].
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean source latent is returned
    /// alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain
    /// img2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep,
        bool keepSourceLatent)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
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

        Tensor t2iNoise = SeedGenerator.CreateNoise(latentShape, seed);
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
}
