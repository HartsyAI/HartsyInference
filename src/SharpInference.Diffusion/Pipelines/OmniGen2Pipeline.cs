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

/// <summary>OmniGen 2 text-to-image pipeline. Caller supplies pre-computed Qwen2.5-VL caption embeddings
/// — OmniGen 2's text encoder is wired through the existing <see cref="Models.TextEncoders.LlamaStyleEncoder"/>
/// at the Qwen2.5-VL preset, but a working Qwen2.5-VL forward + embedding extraction is left to the
/// caller for now. Editing / multi-image-input paths are intentionally out of scope (t2i only).</summary>
public sealed class OmniGen2Pipeline : DiffusionPipelineBase
{
    private readonly OmniGen2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly OmniGen2Config _config;

    /// <summary>Creates an OmniGen 2 t2i pipeline. Caller owns each component.</summary>
    public OmniGen2Pipeline(IBackend backend, OmniGen2Transformer transformer, VaeDecoder vaeDecoder, OmniGen2Config config)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen2.5-VL caption embeddings <c>[1, T, 2048]</c>.
    /// CFG dual-pass when <paramref name="cfgScale"/> &gt; 1 with a negative-prompt embedding.</summary>
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
                "negativeCaptionEmbeddings is required when cfgScale > 1.0.",
                nameof(negativeCaptionEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;

        Logs.Info($"OmniGen 2 t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        FlowMatchEulerDiscreteScheduler scheduler = new(3.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int textSeqLen = (int)captionEmbeddings.Shape[1];

        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights
        // below the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor velocity = _transformer.Forward(Backend, latent, t, captionEmbeddings, textSeqLen);

            if (cfgScale > 1.0f)
            {
                int negSeqLen = (int)negativeCaptionEmbeddings!.Shape[1];
                Tensor uncond = _transformer.Forward(Backend, latent, t, negativeCaptionEmbeddings, negSeqLen);
                Tensor combined = CfgHelper.ApplyCfg(uncond, velocity, cfgScale);
                uncond.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
        latent.Dispose();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"OmniGen 2 t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }
}
