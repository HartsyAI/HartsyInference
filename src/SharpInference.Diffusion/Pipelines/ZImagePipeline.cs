using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Z-Image text-to-image and image-to-image pipeline (Tongyi Lab, Apache 2.0). Accepts pre-computed Qwen3-4B caption embeddings (text-encoder forward is owned by a separate component) and orchestrates the Lumina2/NextDiT transformer with a static-shift flow-match Euler scheduler.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromEmbeddings"/>. Requires a <see cref="VaeEncoder"/> on construction. <b>Caveat:</b> Z-Image's latent normalization is split between this pipeline and the VAE decoder (see <see cref="PrepareVaeInput"/>); the img2img injection point uses the VaeEncoder's standard <c>(raw - shift) * scale</c> output. Strength=0 byte-identical pass-through is exact; nonzero-strength img2img produces structurally-correct output but quality has not been validated against a Python reference. For tight refining quality on Z-Image, prefer the cross-model pixel-space pattern (run a different pipeline as the refiner).</para>
/// </summary>
public sealed unsafe class ZImagePipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ZImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly ZImageConfig _config;
    private int _disposed;

    /// <summary>Creates a Z-Image pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public ZImagePipeline(IBackend backend, ZImageTransformer transformer, VaeDecoder vaeDecoder, ZImageConfig config)
        : this(backend, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Z-Image pipeline with both VAE halves loaded. Required for img2img.</summary>
    public ZImagePipeline(IBackend backend, ZImageTransformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, ZImageConfig config)
    {
        _backend = backend;
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
        Action<GenerationProgress>? onProgress = null)
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
        int width = request.Width;
        int height = request.Height;
        int latentH = height / _config.VaeDownscaleFactor;
        int latentW = width / _config.VaeDownscaleFactor;
        int steps = request.Steps;

        // Img2img validation + strength=0 short-circuit BEFORE any model work.
        int startStep = 0;
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

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), width, height, seed);
            }
        }

        string opMode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "txt2img";
        Logs.Info($"Z-Image {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Static-shift flow-match Euler scheduler ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Build initial latent (t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at sigma[startStep]) ──
        Tensor latent = BuildInitialLatent(request, scheduler, latentShape, seed, startStep);

        // ── 3. Denoising loop (from startStep onward) ──
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Diffusers Z-Image pipeline inverts the timestep: it feeds the transformer
            // `(1 - sigma)` (then the transformer multiplies by t_scale=1000 internally). Without
            // this inversion every step conditions on the OPPOSITE point in the schedule and the
            // model produces near-random output. See pipeline_z_image.py:506.
            float invertedSigma = 1.0f - sigma;
            Tensor velocity = _transformer.Forward(_backend, latent, captionEmbeddings, invertedSigma);

            if (cfgScale > 1.0f)
            {
                Tensor uncondVelocity = _transformer.Forward(_backend, latent, negativeCaptionEmbeddings!, invertedSigma);
                Tensor combined = new Tensor(velocity.Shape, DType.F32);
                ApplyCfg(combined, velocity, uncondVelocity, cfgScale);
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
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        // ── 4. VAE-prep: undo the (latent - shift) * scale that the VAE expects to be inverted ──
        // Z-Image latents are in the same scaled space as Flux: latent_for_vae = latent / scale + shift.
        Tensor vaeInput = PrepareVaeInput(latent, _config.VaeScaleFactor, _config.VaeShiftFactor);
        latent.Dispose();

        // ── 5. VAE decode ──
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, vaeInput);
        vaeInput.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

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
            Tensor sourceLatent = _vaeEncoder!.Encode(_backend, img2img.SourceImage);
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
            _backend.Scale(scaled, t2iNoise, initSigma);
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

    /// <summary>Z-Image CFG combine: <c>combined = pos + cfg * (pos - neg)</c> = <c>(1+cfg)*pos - cfg*neg</c>. NON-STANDARD: most pipelines use <c>uncond + cfg * (cond - uncond)</c>, but Z-Image's diffusers pipeline (<c>pipeline_z_image.py:541</c>) uses pos as the baseline and amplifies the (pos - neg) direction. At cfg=4.0 this gives pred = 5*pos - 4*neg, vs the standard 4*pos - 3*neg — a meaningfully different signal.</summary>
    private static void ApplyCfg(Tensor output, Tensor cond, Tensor uncond, float cfg)
    {
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.Shape.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float c = condPtr[i];
            outPtr[i] = c + cfg * (c - uncondPtr[i]);
        }
    }

    /// <summary>Inverts the VAE's latent normalization: latent_for_vae = latent / scale + shift. The VAE decoder applies (vae_input - shift) * scale internally; here we precompute the left side so the VAE receives raw latents in its native distribution.</summary>
    private static Tensor PrepareVaeInput(Tensor latent, float scaleFactor, float shiftFactor)
    {
        Tensor result = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)result.DataPointer;
        long count = latent.Shape.ElementCount;
        float invScale = 1.0f / scaleFactor;
        for (long i = 0; i < count; i++)
            outPtr[i] = inPtr[i] * invScale + shiftFactor;
        return result;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend, transformer, or VAE — those are shared.</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
