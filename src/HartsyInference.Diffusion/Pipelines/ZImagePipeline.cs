using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial latent = fresh Gaussian noise scaled by initSigma; denoise from step 0).</item>
    /// <item><see cref="ImageToImageRequest"/> → image-to-image. The source image is encoded via the 16-channel Flux/Z-Image VAE and combined with fresh noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// </summary>
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
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0 (Z-Image-Base path). For Z-Image-Turbo, leave cfgScale at 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.ZImageTurbo.Width;
        int height = request.Height ?? GenerationDefaults.ZImageTurbo.Height;
        int latentH = height / _config.VaeDownscaleFactor;
        int latentW = width / _config.VaeDownscaleFactor;
        int steps = request.Steps ?? GenerationDefaults.ZImageTurbo.Steps;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }

        string opMode = isImg2Img ? $"img2img (start={plan.StartStep}/{steps})" : "txt2img";
        Logs.Info($"Z-Image {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Static-shift flow-match Euler scheduler ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Build initial latent (t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at sigma[startStep]) ──
        Tensor latent = BuildInitialLatent(request, scheduler, latentShape, seed, plan.StartStep);

        // ── 3. Denoising loop (from startStep onward) ──
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = plan.StartStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Diffusers Z-Image pipeline inverts the timestep: it feeds the transformer
            // `(1 - sigma)` (then the transformer multiplies by t_scale=1000 internally). Without
            // this inversion every step conditions on the OPPOSITE point in the schedule and the
            // model produces near-random output. See pipeline_z_image.py:506.
            float invertedSigma = 1.0f - sigma;
            Tensor velocity = _transformer.Forward(Backend, latent, captionEmbeddings, invertedSigma, regionalPlan, i - plan.StartStep);

            if (cfgScale > 1.0f)
            {
                Tensor uncondVelocity = _transformer.Forward(Backend, latent, negativeCaptionEmbeddings!, invertedSigma);
                Tensor combined = ApplyZImageCfg(velocity, uncondVelocity, cfgScale);
                uncondVelocity.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            // Diffusers Z-Image pipeline does `noise_pred = -noise_pred` (see pipeline_z_image.py:558).
            // Empirically required: without this we get pure RGB noise; with it we get structured output.
            NegateInPlace(velocity);

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Z-Image step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.ZImage,
            });
        }

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

        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        LogLatentStatsPerChannel("VAE output", image);

        // ── 6. RGB conversion ──
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Z-Image {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: VaeEncoder.Encode(source) combined with fresh noise via flow-matching AddNoise at sigma[startStep].</summary>
    private Tensor BuildInitialLatent(TextToImageRequest request, FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep)
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
            sourceLatent.Dispose();
            noise.Dispose();
            return latent;
        }

        Tensor t2iNoise = SeedGenerator.CreateNoise(latentShape, seed);
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
        Tensor output = new Tensor(cond.Shape, DType.F32);
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.Shape.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float c = condPtr[i];
            outPtr[i] = c + cfg * (c - uncondPtr[i]);
        }
        return output;
    }

    /// <summary>Per-channel min/max/mean diagnostic for a 4D NCHW tensor at Verbose level.
    /// Used to bracket the pre/post VAE state when tracking down all-black output bugs —
    /// healthy Z-Image / Flux latents have per-channel min ~-5 to -1, max ~+1 to +5,
    /// mean within ±2. RGB outputs should land in roughly [-1, 1] with mean near 0.
    /// Outside those bands means the model or VAE saturated.</summary>
    private static void LogLatentStatsPerChannel(string name, Tensor t)
    {
        if (t.Shape.Rank != 4) return;
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
                if (float.IsNaN(v)) { nan++; continue; }
                if (float.IsInfinity(v)) { inf++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            float mean = spatial > 0 ? sum / spatial : 0;
            string flags = nan > 0 || inf > 0 ? $" nan={nan} inf={inf}" : "";
            Logs.Verbose($"  [{name}] ch{c}: min={min:F4} max={max:F4} mean={mean:F4}{flags}");
        }
    }
}
