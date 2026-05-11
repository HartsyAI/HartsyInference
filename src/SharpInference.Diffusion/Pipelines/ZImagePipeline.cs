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
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromEmbeddings"/>. Requires a <see cref="VaeEncoder"/> on construction. Strength=0 byte-identical pass-through is exact; nonzero-strength img2img produces structurally-correct output but quality has not been validated against a Python reference. For tight refining quality on Z-Image, prefer the cross-model pixel-space pattern (run a different pipeline as the refiner). Latent normalization is handled in one place — <see cref="VaeDecoder"/>'s internal UndoScaling — same as Flux.</para>
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
        Tensor image = _vaeDecoder.DecodeTiled(_backend, latent);
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
