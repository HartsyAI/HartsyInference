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

/// <summary>AuraFlow text-to-image pipeline. Uses Pile T5-XXL for text encoding and SDXL-compatible VAE (4-channel latent). Supports standard classifier-free guidance. Flow-matching scheduler with configurable shift.</summary>
public sealed class AuraFlowPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly T5TextEncoder _t5;
    private readonly AuraFlowTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly AuraFlowConfig _config;
    private readonly float _schedulerShift;
    private int _disposed;

    /// <summary>Creates a new AuraFlow pipeline with all components pre-loaded.</summary>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, AuraFlowConfig config, float schedulerShift = 1.73f)
    {
        _backend = backend;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized T5 input with standard CFG.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"AuraFlow: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with T5-XXL ───────────────────────────────────
        Logs.Info("Encoding text with T5-XXL...");

        int[][] batchT5 = [promptTokenIdsT5];
        int[][]? batchMask = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor condContext = _t5.Encode(_backend, batchT5, batchMask);

        Tensor? uncondContext = null;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg)
        {
            int[][] negBatchT5 = [negativePromptTokenIdsT5];
            int[][]? negBatchMask = negativeAttentionMaskT5 is not null ? [negativeAttentionMaskT5] : null;
            uncondContext = _t5.Encode(_backend, negBatchT5, negBatchMask);
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Create initial noise latent [1, 4, latentH, latentW] ─────
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 3. Set up flow-match scheduler ───────────────────────────────
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 4. Denoising loop ────────────────────────────────────────────
        Logs.Info("Starting AuraFlow denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // TODO: Implement denoising step once AuraFlowTransformer.Forward is complete
            // Tensor noisePred;
            // if (useCfg) { run CFG with cond/uncond forward passes }
            // else { single forward pass }
            // Tensor newLatent = scheduler.Step(noisePred, latent, i);

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condContext.Dispose();
        uncondContext?.Dispose();

        // ── 5. VAE decode ────────────────────────────────────────────────
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"AuraFlow image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components.</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
