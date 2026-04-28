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

/// <summary>Z-Image text-to-image pipeline (Tongyi Lab, Apache 2.0). Accepts pre-computed Qwen3-4B caption embeddings (text-encoder forward is owned by a separate component) and orchestrates the Lumina2/NextDiT transformer with a static-shift flow-match Euler scheduler. Turbo (8 NFE, no CFG) is the primary target; Base would add CFG branching but is not yet released.</summary>
public sealed unsafe class ZImagePipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ZImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly ZImageConfig _config;
    private int _disposed;

    /// <summary>Creates a Z-Image pipeline. The transformer and VAE decoder must already have weights loaded (and preferably preloaded onto the backend).</summary>
    public ZImagePipeline(IBackend backend, ZImageTransformer transformer, VaeDecoder vaeDecoder, ZImageConfig config)
    {
        _backend = backend;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen3 caption embeddings.</summary>
    /// <param name="captionEmbeddings">Last-hidden-state output of Qwen3-4B for the prompt [B, capLen, 2560]. The Z-Image system prompt + chat template should already be applied upstream.</param>
    /// <param name="request">Generation parameters (Width, Height, Steps, Seed).</param>
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

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0 (Z-Image-Base path). For Z-Image-Turbo, leave cfgScale at 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / _config.VaeDownscaleFactor;
        int latentW = request.Width / _config.VaeDownscaleFactor;
        int steps = request.Steps;

        Logs.Info($"Z-Image: Generating {request.Width}x{request.Height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Initial noise latent [1, in_channels, latentH, latentW] ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 2. Static-shift flow-match Euler scheduler ──
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 3. Denoising loop ──
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
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
        Logs.Info($"Z-Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>In-place negate. Z-Image's diffusers pipeline negates the velocity output before stepping.</summary>
    private static void NegateInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long count = t.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            p[i] = -p[i];
    }

    /// <summary>combined = uncond + cfg * (cond - uncond) — standard CFG combine.</summary>
    private static void ApplyCfg(Tensor output, Tensor cond, Tensor uncond, float cfg)
    {
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.Shape.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float u = uncondPtr[i];
            outPtr[i] = u + cfg * (condPtr[i] - u);
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
