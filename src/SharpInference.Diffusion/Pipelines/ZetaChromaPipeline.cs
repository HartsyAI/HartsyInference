using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Zeta-Chroma text-to-image pipeline (<c>lodestones/Zeta-Chroma</c> pixel-proto) — pixel-space, VAE-free.
/// Accepts pre-computed Qwen3-4B caption embeddings (same upstream encoder path as <see cref="ZImagePipeline"/>)
/// and runs the <see cref="ZetaChromaTransformer"/> directly on RGB in [-1, 1].
///
/// Sampling (ALL validation-gated — the model is mid-pretraining, see docs/Research/CHROMA_RADIANCE_ARCHITECTURE.md
/// §Zeta-Chroma):
/// <list type="bullet">
///   <item><b>x0 prediction</b> — converted per step via <see cref="X0Prediction.ToVelocity"/>
///         (<c>v = (x_t − x0) / t</c>, ComfyUI <c>NextDiTPixelSpace</c>), then standard CFG on v
///         (<see cref="CfgHelper.ApplyCfg"/> — NOT Z-Image's non-standard cond-baseline formula; item 8).</item>
///   <item><b>Timestep inversion</b> — the transformer is conditioned on <c>1 − sigma</c>, inheriting Z-Image's
///         convention (item 7).</item>
///   <item><b>Flow-match Euler</b>, static shift 3.0 (item 9), default 50 steps, CFG 5.0.</item>
///   <item><b>Resolution must be divisible by the pixel patch size</b> (32 reported) — no pad/crop path until the
///         training resolution behavior is known.</item>
/// </list></summary>
public sealed class ZetaChromaPipeline : DiffusionPipelineBase
{
    private readonly ZetaChromaTransformer _transformer;
    private readonly ZetaChromaConfig _config;

    /// <summary>Creates a Zeta-Chroma pipeline. The Qwen3 caption encoder is owned by the caller (as with Z-Image).</summary>
    public ZetaChromaPipeline(IBackend backend, ZetaChromaTransformer transformer, ZetaChromaConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen3 caption embeddings. API mirrors
    /// <see cref="ZImagePipeline.GenerateFromEmbeddings"/> (txt2img only — no img2img until validated).</summary>
    /// <param name="captionEmbeddings">Qwen3-4B last-hidden-state for the prompt [B, capLen, 2560].</param>
    /// <param name="request">Generation parameters. Width/Height must be divisible by the 32-px patch.</param>
    /// <param name="cfgScale">CFG scale (5.0 recommended; 1.0 disables the second pass).</param>
    /// <param name="negativeCaptionEmbeddings">Negative-prompt embeddings, required when <paramref name="cfgScale"/> &gt; 1.</param>
    /// <param name="onProgress">Optional progress callback per step.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor captionEmbeddings,
        TextToImageRequest request,
        float cfgScale = 5.0f,
        Tensor? negativeCaptionEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (cfgScale > 1.0f && negativeCaptionEmbeddings is null)
            throw new ArgumentException(
                "negativeCaptionEmbeddings is required when cfgScale > 1.0.", nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int steps = request.Steps;
        int patch = _transformer.PatchSize;
        bool useCfg = cfgScale > 1.0f;

        if (width % patch != 0 || height % patch != 0)
            throw new ArgumentException(
                $"Zeta-Chroma is pixel-space with a {patch}-px patch; width/height ({width}x{height}) must be " +
                $"divisible by {patch}.", nameof(request));

        Logs.Info($"Zeta-Chroma: Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed} " +
            "(mid-pretraining checkpoint — output quality is validation-gated)");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Static-shift flow-match Euler scheduler ──
        TensorShape pixelShape = new TensorShape(1, 3, height, width);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Initial pixel sample: pure noise scaled by initSigma ──
        Tensor pixels = request.InitialNoise ?? SeedGenerator.CreateNoise(pixelShape, seed);
        if (!pixels.Shape.Equals(pixelShape))
            throw new ArgumentException($"InitialNoise shape {pixels.Shape} != expected {pixelShape}.");
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(pixelShape, DType.F32);
            Backend.Scale(scaled, pixels, initSigma);
            pixels.Dispose();
            pixels = scaled;
        }

        // ── 3. Denoising loop ──
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Z-Image lineage: the transformer conditions on the INVERTED timestep (1 - sigma); the
            // transformer multiplies by t_scale=1000 internally. Validation-gated (research doc item 7).
            float invertedSigma = 1.0f - sigma;

            Tensor condX0 = _transformer.Forward(Backend, pixels, captionEmbeddings, invertedSigma);
            Tensor velocity = X0Prediction.ToVelocity(condX0, pixels, sigma);
            condX0.Dispose();

            if (useCfg)
            {
                Tensor uncondX0 = _transformer.Forward(Backend, pixels, negativeCaptionEmbeddings!, invertedSigma);
                Tensor uncondV = X0Prediction.ToVelocity(uncondX0, pixels, sigma);
                uncondX0.Dispose();

                // Standard CFG on velocity — assumed (validation-gated, item 8); deliberately NOT
                // ZImagePipeline's cond-baseline variant.
                Tensor combined = CfgHelper.ApplyCfg(uncondV, velocity, cfgScale);
                uncondV.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            Tensor newPixels = new Tensor(pixelShape, DType.F32);
            scheduler.Step(newPixels, velocity, pixels, i);
            velocity.Dispose();
            pixels.Dispose();
            pixels = newPixels;

            stepSw.Stop();
            Logs.Debug($"Zeta-Chroma step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = pixels,
                LatentArch = LatentArchitecture.ZetaChroma,
            });
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // ── 4. Direct RGB conversion — no VAE ──
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(pixels);
        pixels.Dispose();

        sw.Stop();
        Logs.Info($"Zeta-Chroma generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }
}
