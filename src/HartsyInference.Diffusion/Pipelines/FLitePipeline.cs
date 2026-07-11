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

    // Prompt-embedding cache (the Flux2/Krea2 pattern): a hit skips the whole T5 phase, which also
    // avoids the TE⇄DiT VRAM swap. Keyed on the positive tokens + whether a real negative was encoded.
    private int[]? _cachedPromptKey;
    private Tensor? _cachedPositive;
    private Tensor? _cachedNegative;
    private bool _ditResident;
    private readonly FLiteTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly FLiteConfig _config;
    private readonly float _vaeScalingFactor;
    private readonly float _vaeShiftFactor;

    /// <summary>Creates an F-Lite text-to-image pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="vaeScalingFactor">Flux Schnell VAE scaling factor (default 0.3611). Latent is divided by this before VAE decode.</param>
    /// <param name="vaeShiftFactor">Flux Schnell VAE shift factor (default 0.1159). Added back to latent before VAE decode.</param>
    public FLitePipeline(IBackend backend, T5TextEncoder t5, FLiteTransformer transformer,
        VaeDecoder vaeDecoder, FLiteConfig config,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : this(backend, t5, transformer, vaeDecoder, vaeEncoder: null, config, vaeScalingFactor, vaeShiftFactor)
    {
    }

    /// <summary>Creates an F-Lite pipeline with both VAE halves loaded — required for img2img / inpaint (pass an
    /// <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Configure the encoder with
    /// <see cref="VaeConfig.Flux"/> so its output is already normalized (<c>(mu − shift) · scale</c>) into the
    /// transformer's latent space — the inverse of the <c>latent / scale + shift</c> this pipeline applies before
    /// decode.</summary>
    public FLitePipeline(IBackend backend, T5TextEncoder t5, FLiteTransformer transformer,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, FLiteConfig config,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _vaeScalingFactor = vaeScalingFactor;
        _vaeShiftFactor = vaeShiftFactor;
    }

    /// <summary>Generates an image from pre-tokenized T5 input. <paramref name="t5TokenIds"/> length determines the conditioning sequence length (F-Lite reference uses 512 with padding).
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source is VAE-encoded (Flux Schnell VAE) and
    /// mixed with fresh noise at F-Lite's dynamic-shift time <c>t(startStep)</c> (<c>x = (1−t)·src + t·noise</c>, the
    /// same convention the inline integrator denoises under) — requires a <see cref="VaeEncoder"/> on construction.
    /// A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step latent blend + final pixel recomposite).
    /// Strength=0 short-circuits to byte-identical pass-through.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] t5TokenIds, int[] t5AttentionMask,
        int[]? negativeT5TokenIds, int[]? negativeT5AttentionMask,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.FLite.Width;
        int height = request.Height ?? GenerationDefaults.FLite.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "txt2img";
        Logs.Info($"F-Lite {opMode}: {width}x{height} (latent {latentH}x{latentW}), {steps} steps, cfg={cfgScale}, seed={seed}.");
        Stopwatch totalSw = Stopwatch.StartNew();

        Stopwatch sw = Stopwatch.StartNew();
        // Bulk-upload T5 weights once so the encoder's many kernel launches don't each pay a
        // per-op cache-miss H2D transfer. Paired with FreeWeights immediately after encoding
        // since T5-XXL is ~5 GB and we want it gone before the transformer phase. No-op on
        // backends without a weight cache.
        // Cache covers the common zero-negative path only; a real negative always re-encodes.
        bool cacheHit = negativeT5TokenIds is null && _cachedPromptKey is not null
            && t5TokenIds.AsSpan().SequenceEqual(_cachedPromptKey);
        Tensor positiveContext;
        Tensor negativeContext;
        if (cacheHit)
        {
            positiveContext = _cachedPositive!;
            negativeContext = _cachedNegative!;
            Logs.Info("F-Lite prompt cache HIT — skipping T5 encode.");
            goto AfterEncode;
        }
        // Cache miss: the 20GB DiT cannot share the card with T5-XXL's preload — evict it first.
        if (_ditResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        Backend.TrimMemoryPool();
        Backend.PreloadWeights(_t5.EnumerateWeights());
        int[][] posTokens = [t5TokenIds];
        int[][] posMask = [t5AttentionMask];
        positiveContext = _t5.EncodeAtLayer(Backend, posTokens, _config.T5LayerIndex, applyFinalNorm: true, posMask);
        // fal-ai reference: an unset negative prompt is a ZERO context tensor (torch.zeros_like), not an
        // encoded empty string.
        if (negativeT5TokenIds is null)
        {
            negativeContext = new Tensor(positiveContext.Shape, DType.F32);
            Backend.Fill(negativeContext, 0.0f);
        }
        else
        {
            int[][] negTokens = [negativeT5TokenIds];
            int[][] negMask = [negativeT5AttentionMask!];
            negativeContext = _t5.EncodeAtLayer(Backend, negTokens, _config.T5LayerIndex, applyFinalNorm: true, negMask);
        }
        sw.Stop();
        Logs.Info($"T5 encode (layer {_config.T5LayerIndex}) done in {sw.ElapsedMilliseconds}ms (pos shape={positiveContext.Shape}, neg shape={negativeContext.Shape}).");

        Backend.Sync();
        Backend.FreeWeights(_t5.EnumerateWeights());
        // Host-materialize so the cached embeddings survive activation sweeps, then cache for reuse.
        unsafe { _ = positiveContext.DataPointer; _ = negativeContext.DataPointer; }
        Backend.FreeActivations();
        Backend.TrimMemoryPool();
        if (!ReferenceEquals(_cachedPositive, positiveContext)) _cachedPositive?.Dispose();
        if (!ReferenceEquals(_cachedNegative, negativeContext)) _cachedNegative?.Dispose();
        _cachedPositive = positiveContext;
        _cachedNegative = negativeContext;
        _cachedPromptKey = (int[])t5TokenIds.Clone();
        AfterEncode: ;

        // Reference alpha uses the LATENT cell grid (h/8 · w/8), not the post-patchify token grid:
        // fal-ai pipeline.py `image_token_size = latent_height * latent_width` → alpha 4 at 1024².
        int imgTokenCount = latentH * latentW;
        float alpha = 2.0f * MathF.Sqrt(imgTokenCount / (64.0f * 64.0f));
        Logs.Info($"F-Lite: dynamic-shift alpha={alpha:F3} for {hPacked}x{wPacked} image-token grid.");

        // Initial latent. T2I: pure noise (caller-injected or seeded) at t(0)=1. Img2img: VAE-encoded source
        // mixed with fresh noise at the shifted start time — x = (1−t)·src + t·noise with t = t(startStep),
        // matching the integrator's x_t = (1−t)·x0 + t·ε convention (t: 1 → 0 over the loop).
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        Tensor latent;
        Tensor? sourceLatent = null;
        Tensor? latentMask = null;
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor encoded = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            float tStart = ShiftedTime(startStep, steps, alpha);
            latent = new Tensor(latentShape, DType.F32);
            Img2ImgSetup.MixAtSigma(latent, encoded, noise, tStart);
            noise.Dispose();
            if (isMaskedInpaint)
            {
                sourceLatent = encoded;
                latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
            }
            else
            {
                encoded.Dispose();
            }
        }
        else
        {
            latent = request.InitialNoise ?? SeedGenerator.CreateNoise(latentShape, seed);
            if (!latent.Shape.Equals(latentShape) || latent.DType != DType.F32)
                throw new ArgumentException($"InitialNoise must be F32 with shape {latentShape}; got {latent.Shape} {latent.DType}.", nameof(request));
        }

        // Bulk-upload transformer weights before the denoise loop. F-Lite is a 40-block
        // single-stream DiT — without preload the first step would pay cache-miss overhead
        // for every block. Paired with FreeWeights below the VAE handoff.
        if (!_ditResident)
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
        }

        Tensor accumulator = new Tensor(latentShape, DType.F32);
        Buffer.MemoryCopy((void*)latent.DataPointer, (void*)accumulator.DataPointer,
            (long)latent.ElementCount * sizeof(float),
            (long)latent.ElementCount * sizeof(float));

        bool useCfg = cfgScale > 1.0f;
        for (int step = startStep; step < steps; step++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = ShiftedTime(step, steps, alpha);
            float tNext = ShiftedTime(step + 1, steps, alpha);
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

            // Masked-inpaint blend directly on the accumulator (the integrator's source of truth): keep the
            // unmasked region on the source's trajectory by re-mixing the source with fresh noise at tNext.
            // Final step blends with the clean source (tNext ≈ 0) — no further integration follows.
            if (latentMask is not null && sourceLatent is not null)
            {
                Tensor noisedSource;
                if (step + 1 < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(latentShape, seed + step + 1);
                    noisedSource = new Tensor(latentShape, DType.F32);
                    Img2ImgSetup.MixAtSigma(noisedSource, sourceLatent, freshNoise, tNext);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourceLatent;
                }
                MaskBlendUtilities.BlendChannelsInPlace(accumulator, noisedSource, latentMask);
                if (noisedSource != sourceLatent) noisedSource.Dispose();
            }

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
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        Backend.Sync();

        ApplyVaeShiftScale(latent);
        Logs.Info("F-Lite VAE decode...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decoded = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms.");

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(decoded, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        totalSw.Stop();
        Logs.Info($"F-Lite {opMode} total: {totalSw.ElapsedMilliseconds}ms (seed={seed}).");
        return (rgbData, width, height, seed);
    }

    /// <summary>F-Lite's dynamic-shift time at loop index <paramref name="stepIndex"/>: <c>t = shift((steps−i)/steps)</c>
    /// with <c>shift(u) = u·alpha / (1 + (alpha−1)·u)</c>. t(0) = 1 (pure noise), t(steps) = 0 (clean).</summary>
    private static float ShiftedTime(int stepIndex, int steps, float alpha)
    {
        float tNorm = (steps - stepIndex) / (float)steps;
        return tNorm * alpha / (1.0f + (alpha - 1.0f) * tNorm);
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
