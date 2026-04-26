using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Hunyuan Image 2.1 text-to-image pipeline. Uses dual text encoders (CLIP-like + T5/mT5), 32-channel VAE with 32× downscale, and flow-match scheduling. Supports native 2048×2048 resolution and distilled variants.</summary>
public sealed class HunyuanImagePipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipEncoder;
    private readonly T5TextEncoder _t5Encoder;
    private readonly HunyuanImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly HunyuanImageConfig _config;
    private int _disposed;

    /// <summary>Creates a new Hunyuan Image pipeline.</summary>
    public HunyuanImagePipeline(IBackend backend, ClipTextEncoder clipEncoder, T5TextEncoder t5Encoder,
        HunyuanImageTransformer transformer, VaeDecoder vaeDecoder, HunyuanImageConfig config)
    {
        _backend = backend;
        _clipEncoder = clipEncoder;
        _t5Encoder = t5Encoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized input.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsClip,
        int[] negativePromptTokenIdsClip,
        int promptEosPositionClip,
        int negativeEosPositionClip,
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        // 32× downsample: 2048 → 64 latent spatial
        int latentH = request.Height / 32;
        int latentW = request.Width / 32;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"Hunyuan Image 2.1: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text ───────────────────────────────────────────────
        // TODO: Implement dual text encoding (CLIP-like pooled + T5/mT5 per-token)
        // - CLIP: produce pooled embedding for AdaLN conditioning
        // - T5/mT5: produce per-token context for cross-attention
        Logs.Info("Encoding text with CLIP + T5...");

        // ── 2. Create initial noise [1, 32, latentH, latentW] ────────────
        TensorShape latentShape = new TensorShape(1, 32, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 3. Set up scheduler ──────────────────────────────────────────
        // TODO: Determine correct scheduler shift for Hunyuan Image
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(3.0f);
        scheduler.SetTimesteps(steps);

        // ── 4. Denoising loop ────────────────────────────────────────────
        Logs.Info("Starting Hunyuan Image denoising loop...");
        // TODO: Implement denoising loop once transformer forward pass is complete

        // ── 5. VAE decode ────────────────────────────────────────────────
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        // TODO: VAE decode with 32-channel Hunyuan VAE
        // Tensor image = _vaeDecoder.Decode(_backend, latent);
        vaeSw.Stop();

        latent.Dispose();
        sw.Stop();
        Logs.Info($"Hunyuan Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        // TODO: Return actual image data once pipeline is fully implemented
        throw new NotImplementedException("HunyuanImagePipeline requires transformer and VAE implementation");
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
