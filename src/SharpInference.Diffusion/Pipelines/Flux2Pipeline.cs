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

/// <summary>Flux.2 text-to-image pipeline. Similar to Flux.1 but with 16×16 VAE, Mistral/Qwen text encoder, and evolved transformer architecture. Core transformer reuses FluxTransformer with Flux2Config-derived FluxConfig (no QKV bias, different block counts). Requires FP8 for 32B Dev on consumer GPUs.</summary>
public sealed unsafe class Flux2Pipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly FluxTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Flux2Config _config;
    private int _disposed;

    // Text encoder is variant-dependent (Mistral or Qwen) — stored as opaque reference
    // TODO: Define ITextEncoder interface or use concrete Mistral/Qwen encoder classes
    // For now, the pipeline accepts pre-computed text embeddings

    /// <summary>Creates a new Flux.2 pipeline.</summary>
    public Flux2Pipeline(IBackend backend, FluxTransformer transformer, VaeDecoder vaeDecoder, Flux2Config config)
    {
        _backend = backend;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed text embeddings.</summary>
    /// <param name="textEmbeddings">Per-token text embeddings [B, seqLen, contextDim] from Mistral/Qwen.</param>
    /// <param name="pooledEmbedding">Pooled text embedding [B, vecInDim].</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="guidanceScale">Guidance scale for embedded guidance.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor textEmbeddings,
        Tensor pooledEmbedding,
        TextToImageRequest request,
        float guidanceScale = 3.5f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        // Flux.2 uses 16× downsample VAE
        int latentH = request.Height / _config.VaeDownscaleFactor;
        int latentW = request.Width / _config.VaeDownscaleFactor;
        int steps = request.Steps;

        string variant = _config.TextEncoderType == Flux2TextEncoderType.Mistral ? "Dev 32B" : "Klein";
        Logs.Info($"Flux.2 ({variant}): Generating {request.Width}x{request.Height} image, {steps} steps, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Create initial noise latent ───────────────────────────────
        // TODO: Determine correct latent channel count for Flux.2 16×16 VAE
        int latentChannels = 16;
        TensorShape latentShape = new TensorShape(1, latentChannels, latentH, latentW);
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);

        // Pack latent for transformer
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;

        // TODO: PackLatent for Flux.2 (may differ from Flux.1 due to 16× VAE)

        // ── 2. Set up flow-match scheduler ───────────────────────────────
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // ── 3. Denoising loop ────────────────────────────────────────────
        Logs.Info("Starting Flux.2 denoising loop...");
        // TODO: Implement denoising loop using FluxTransformer with Flux.2 config
        // The transformer is architecturally the same but with different config

        // ── 4. VAE decode ────────────────────────────────────────────────
        // TODO: VAE decode with Flux.2 16×16 VAE

        noise.Dispose();
        sw.Stop();
        Logs.Info($"Flux.2 image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        throw new NotImplementedException("Flux2Pipeline requires Flux.2 VAE, text encoder integration, and architecture verification");
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline.</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
