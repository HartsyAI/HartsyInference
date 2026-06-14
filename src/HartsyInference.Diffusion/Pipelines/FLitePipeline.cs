using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>F-Lite pipeline (Freepik / Fal.ai). T5-XXL layer-17 conditioning + 16-channel Flux Schnell VAE + 40-block single-stream cross-attention DiT with V-residual. CFG via batch-of-2 forward (single transformer call instead of dual-pass — saves a forward at the cost of 2× peak activation memory).
///
/// <para><b>Inline scheduler</b>: F-Lite uses a custom dynamic-shift flow-match integrator with <c>alpha = 2 * sqrt(image_token_count / (64 * 64))</c>. Per the reference pipeline, the loop is implemented inline rather than through an <see cref="HartsyInference.Core.Schedulers.IScheduler"/> — this avoids a 3-line scheduler shell and keeps the integration close to the reference for first-run debugging.</para>
///
/// <para><b>Status (2026-05-06)</b>: implementation tracked against [`F_LITE_ARCHITECTURE.md`](../../../docs/Research/F_LITE_ARCHITECTURE.md). End-to-end visual validation against the actual `Freepik/F-Lite` checkpoint is pending download. Expect 1-3 first-run bugs to surface (typical for a new model port; see SD3.5 / Z-Image debug histories in PHASE_3_DEVIATIONS.md).</para></summary>
public sealed unsafe class FLitePipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly FLiteTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly FLiteConfig _config;
    private readonly float _vaeScalingFactor;
    private readonly float _vaeShiftFactor;

    /// <summary>Creates an F-Lite text-to-image pipeline.</summary>
    /// <param name="vaeScalingFactor">Flux Schnell VAE scaling factor (default 0.3611). Latent is divided by this before VAE decode.</param>
    /// <param name="vaeShiftFactor">Flux Schnell VAE shift factor (default 0.1159). Added back to latent before VAE decode.</param>
    public FLitePipeline(IBackend backend, T5TextEncoder t5, FLiteTransformer transformer,
        VaeDecoder vaeDecoder, FLiteConfig config,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _vaeScalingFactor = vaeScalingFactor;
        _vaeShiftFactor = vaeShiftFactor;
    }

    /// <summary>Generates an image from pre-tokenized T5 input. <paramref name="t5TokenIds"/> length determines the conditioning sequence length (F-Lite reference uses 512 with padding).</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] t5TokenIds, int[] t5AttentionMask,
        int[] negativeT5TokenIds, int[] negativeT5AttentionMask,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"F-Lite: {width}x{height} (latent {latentH}x{latentW}), {steps} steps, cfg={cfgScale}, seed={seed}.");
        Stopwatch totalSw = Stopwatch.StartNew();

        Stopwatch sw = Stopwatch.StartNew();
        // Bulk-upload T5 weights once so the encoder's many kernel launches don't each pay a
        // per-op cache-miss H2D transfer. Paired with FreeWeights immediately after encoding
        // since T5-XXL is ~5 GB and we want it gone before the transformer phase. No-op on
        // backends without a weight cache.
        Backend.PreloadWeights(_t5.EnumerateWeights());
        int[][] posTokens = [t5TokenIds];
        int[][] posMask = [t5AttentionMask];
        Tensor positiveContext = _t5.EncodeAtLayer(Backend, posTokens, _config.T5LayerIndex, applyFinalNorm: true, posMask);
        int[][] negTokens = [negativeT5TokenIds];
        int[][] negMask = [negativeT5AttentionMask];
        Tensor negativeContext = _t5.EncodeAtLayer(Backend, negTokens, _config.T5LayerIndex, applyFinalNorm: true, negMask);
        sw.Stop();
        Logs.Info($"T5 encode (layer {_config.T5LayerIndex}) done in {sw.ElapsedMilliseconds}ms (pos shape={positiveContext.Shape}, neg shape={negativeContext.Shape}).");

        Backend.Sync();
        Backend.FreeWeights(_t5.EnumerateWeights());

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        Tensor latent = request.InitialNoise ?? SeedGenerator.CreateNoise(latentShape, seed);
        if (!latent.Shape.Equals(latentShape) || latent.DType != DType.F32)
            throw new ArgumentException($"InitialNoise must be F32 with shape {latentShape}; got {latent.Shape} {latent.DType}.", nameof(request));

        int imgTokenCount = hPacked * wPacked;
        float alpha = 2.0f * MathF.Sqrt(imgTokenCount / (64.0f * 64.0f));
        Logs.Info($"F-Lite: dynamic-shift alpha={alpha:F3} for {hPacked}x{wPacked} image-token grid.");

        // Bulk-upload transformer weights before the denoise loop. F-Lite is a 40-block
        // single-stream DiT — without preload the first step would pay cache-miss overhead
        // for every block. Paired with FreeWeights below the VAE handoff.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Tensor accumulator = new Tensor(latentShape, DType.F32);
        Buffer.MemoryCopy((void*)latent.DataPointer, (void*)accumulator.DataPointer,
            (long)latent.ElementCount * sizeof(float),
            (long)latent.ElementCount * sizeof(float));

        bool useCfg = cfgScale > 1.0f;
        for (int step = 0; step < steps; step++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            int idx = steps - step;
            float tNorm = idx / (float)steps;
            float tNextNorm = (idx - 1) / (float)steps;
            float t = tNorm * alpha / (1.0f + (alpha - 1.0f) * tNorm);
            float tNext = tNextNorm * alpha / (1.0f + (alpha - 1.0f) * tNextNorm);
            float dt = t - tNext;

            Tensor velocity;
            if (useCfg)
            {
                Tensor uncond = _transformer.Forward(Backend, latent, negativeContext, t);
                Tensor cond = _transformer.Forward(Backend, latent, positiveContext, t);
                velocity = CfgHelper.ApplyCfg(uncond, cond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                velocity = _transformer.Forward(Backend, latent, positiveContext, t);
            }

            ApplyEulerStepInPlace(accumulator, velocity, dt);
            velocity.Dispose();

            latent.Dispose();
            latent = new Tensor(latentShape, DType.F32);
            Buffer.MemoryCopy((void*)accumulator.DataPointer, (void*)latent.DataPointer,
                (long)accumulator.ElementCount * sizeof(float),
                (long)accumulator.ElementCount * sizeof(float));

            stepSw.Stop();
            Logs.Info($"F-Lite step {step + 1}/{steps} (t={t:F3} → {tNext:F3}, dt={dt:F3}) done in {stepSw.ElapsedMilliseconds}ms.");
            onProgress?.Invoke(new GenerationProgress(step + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.FLite,
            });
        }
        accumulator.Dispose();
        positiveContext.Dispose();
        negativeContext.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        ApplyVaeShiftScale(latent);
        Logs.Info("F-Lite VAE decode...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decoded = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms.");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        totalSw.Stop();
        Logs.Info($"F-Lite total: {totalSw.ElapsedMilliseconds}ms (seed={seed}).");
        return (rgbData, width, height, seed);
    }

    private static void ApplyEulerStepInPlace(Tensor accumulator, Tensor velocity, float dt)
    {
        float* accPtr = (float*)accumulator.DataPointer;
        float* velPtr = (float*)velocity.DataPointer;
        long count = accumulator.ElementCount;
        for (long i = 0; i < count; i++)
        {
            accPtr[i] += dt * velPtr[i];
        }
    }

    private void ApplyVaeShiftScale(Tensor latent)
    {
        float* p = (float*)latent.DataPointer;
        long count = latent.ElementCount;
        float invScale = 1.0f / _vaeScalingFactor;
        for (long i = 0; i < count; i++)
        {
            p[i] = p[i] * invScale + _vaeShiftFactor;
        }
    }
}
