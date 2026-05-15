using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Models.Vae.QwenImage;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Anima (Cosmos-Predict2 family) text-to-image pipeline. Image-only path with two-stage text conditioning:
/// caller-supplied Qwen-3 0.6B hidden states are first refined by the in-checkpoint
/// <see cref="AnimaLlmAdapter"/> (a 6-block self+cross+MLP transformer) into 1024-dim features, which then feed
/// the DiT's cross-attention. The flow-match Euler scheduler is fixed at <c>shift = 3.0</c> per the Anima reference
/// workflow (ER-SDE substitute; pixel parity with Comfy's <c>er_sde + simple</c> requires a dedicated SDE scheduler
/// — see roadmap).</summary>
public sealed unsafe class AnimaPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly AnimaTransformer _transformer;
    private readonly AnimaLlmAdapter _llmAdapter;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly AnimaConfig _config;
    private int _disposed;

    /// <summary>Creates an Anima t2i pipeline. Caller owns the components. The VAE is the
    /// <see cref="QwenImageVaeDecoder"/> (3D causal autoencoder collapsed to 2D for image mode);
    /// the standard <see cref="VaeDecoder"/> class can't load this checkpoint because the key
    /// layout and conv ranks differ.</summary>
    public AnimaPipeline(IBackend backend, AnimaTransformer transformer, AnimaLlmAdapter llmAdapter,
        QwenImageVaeDecoder vaeDecoder, AnimaConfig config)
    {
        _backend = backend;
        _transformer = transformer;
        _llmAdapter = llmAdapter;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen-3 0.6B hidden states <c>[1, T, 1024]</c>. The pipeline
    /// runs them through the LlmAdapter to produce refined features, then through the DiT cross-attention.
    /// CFG dual-pass when <paramref name="cfgScale"/> &gt; 1 and a negative-prompt embedding is provided.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor textEmbeddings,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeTextEmbeddings = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (cfgScale > 1.0f && negativeTextEmbeddings is null)
            throw new ArgumentException(
                "negativeTextEmbeddings is required when cfgScale > 1.0.",
                nameof(negativeTextEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;

        Logs.Info($"Anima t2i: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        LogStats("textEmbeddings (Qwen3 0.6B output)", textEmbeddings);

        // Diagnostic toggle: ANIMA_BYPASS_LLM_ADAPTER=1 skips the in-checkpoint LlmAdapter and feeds raw
        // Qwen-3 hidden states directly to the DiT cross-attention. The DiT's cross_attn.{k,v}_proj weights
        // are [2048, 1024] and Qwen-3 outputs 1024-dim features, so the projection is geometrically valid
        // either way. If output becomes coherent with the bypass, the LlmAdapter implementation is the
        // dominant issue and we need to revisit its forward (e.g., cross-attn K/V source) per the
        // PHASE_3 troubleshooting methodology.
        bool bypassAdapter = Environment.GetEnvironmentVariable("ANIMA_BYPASS_LLM_ADAPTER") == "1";
        Tensor refinedText;
        if (bypassAdapter)
        {
            Logs.Info("[Anima] ANIMA_BYPASS_LLM_ADAPTER=1 — feeding raw Qwen-3 features to DiT cross-attn (diagnostic mode).");
            refinedText = textEmbeddings.DType == DType.F32 ? CloneF32(textEmbeddings) : textEmbeddings.CastTo(DType.F32);
        }
        else
        {
            // Refine the Qwen-3 0.6B hidden states through the LlmAdapter ONCE per generation (timestep-independent).
            refinedText = _llmAdapter.Forward(_backend, textEmbeddings);
        }
        LogStats("refinedText (LlmAdapter output)", refinedText);

        Tensor? refinedNegText = null;
        if (negativeTextEmbeddings is not null)
        {
            refinedNegText = bypassAdapter
                ? (negativeTextEmbeddings.DType == DType.F32 ? CloneF32(negativeTextEmbeddings) : negativeTextEmbeddings.CastTo(DType.F32))
                : _llmAdapter.Forward(_backend, negativeTextEmbeddings);
        }

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);
        LogStats("initial noise latent", latent);

        FlowMatchEulerDiscreteScheduler scheduler = new(3.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // CRITICAL: Cosmos / Anima expects timestep in [0, ~1] range, NOT [0, 1000] (SD3/Flux convention).
        // Per diffusers' pipeline_cosmos2_text2image.py line 588-599, the transformer receives
        // `current_t = sigma / (sigma + 1)` directly — no ×1000 scaling. SharpInference's
        // FlowMatchEulerDiscreteScheduler stores `_timesteps[i] = sigma * 1000` for SD3 compatibility,
        // so we divide by 1000 here to get the Cosmos-convention timestep. Without this, the model's
        // sinusoidal time embedding sees values 1000× larger than training distribution, AdaLN
        // modulation produces near-random shift/scale, and the velocity has no consistent denoising
        // direction.
        Logs.Info($"Anima timesteps (after /1000 normalization): [{timesteps[0] / 1000f:F4}, {timesteps[1] / 1000f:F4}, ..., {timesteps[^1] / 1000f:F4}] (expected range ~[1.0, 0.0])");

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i] / 1000f;  // Cosmos-convention timestep in [0, ~1].

            Tensor velocity = _transformer.Forward(_backend, latent, t, refinedText);

            if (cfgScale > 1.0f)
            {
                Tensor uncond = _transformer.Forward(_backend, latent, t, refinedNegText!);
                Tensor combined = new(velocity.Shape, DType.F32);
                ApplyStandardCfg(combined, velocity, uncond, cfgScale);
                uncond.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            if (i == 0 || i == steps / 2 || i == steps - 1)
                LogStats($"velocity step={i} (t={t:F1})", velocity);

            Tensor newLatent = new(latentShape, DType.F32);
            scheduler.Step(newLatent, velocity, latent, i);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        refinedText.Dispose();
        refinedNegText?.Dispose();

        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());
        _backend.FreeWeights(_llmAdapter.EnumerateWeights());

        LogStats("final latent (pre-VAE)", latent);
        LogPerChannelStats("final latent (pre-VAE) per-channel", latent);

        Tensor decoded = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();

        LogStats("VAE decoded (raw)", decoded);
        LogPerChannelStats("VAE decoded per-channel", decoded);

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        sw.Stop();
        Logs.Info($"Anima t2i complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    /// <summary>Print min/max/mean/abs_mean/std/has_nan/has_inf for a tensor, F32 or convertible. Used to
    /// trace where the pipeline diverges from expected ranges per the Phase 3/4 debugging methodology
    /// (see PHASE_3_DEVIATIONS.md — the recurring failure mode is "structurally right but numerically off",
    /// and the cheapest diagnostic is stats prints at every boundary).</summary>
    private static unsafe void LogStats(string label, Tensor t)
    {
        Tensor src = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        long count = src.Shape.ElementCount;
        float* p = (float*)src.DataPointer;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        double sum = 0, sumSq = 0, sumAbs = 0;
        long nanCount = 0, infCount = 0;
        for (long i = 0; i < count; i++)
        {
            float v = p[i];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSq += (double)v * v;
            sumAbs += Math.Abs(v);
        }
        long finite = count - nanCount - infCount;
        double mean = finite > 0 ? sum / finite : 0;
        double absMean = finite > 0 ? sumAbs / finite : 0;
        double std = finite > 0 ? Math.Sqrt(Math.Max(0, sumSq / finite - mean * mean)) : 0;
        string nanInf = (nanCount > 0 || infCount > 0) ? $"  *** NAN={nanCount} INF={infCount} ***" : "";
        Logs.Info($"[Anima stats] {label,-44} shape={src.Shape}  min={min:F3}  max={max:F3}  mean={mean:F4}  abs_mean={absMean:F4}  std={std:F4}{nanInf}");
        if (!ReferenceEquals(src, t)) src.Dispose();
    }

    /// <summary>Print per-channel mean/std/abs_mean of a 4-D tensor [B, C, H, W]. Used to detect whether
    /// individual latent channels carry distinct content (suggesting the DiT produced meaningful output
    /// and the VAE is the issue) vs all channels looking statistically identical (suggesting the DiT
    /// is producing uniform-ish noise per channel = DiT is the issue).</summary>
    private static unsafe void LogPerChannelStats(string label, Tensor t)
    {
        Tensor src = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        if (src.Shape.Rank != 4)
        {
            Logs.Info($"[Anima per-ch] {label} — not 4D, shape={src.Shape}, skipping.");
            if (!ReferenceEquals(src, t)) src.Dispose();
            return;
        }
        int batch = (int)src.Shape[0];
        int channels = (int)src.Shape[1];
        int h = (int)src.Shape[2];
        int w = (int)src.Shape[3];
        long spatial = (long)h * w;
        float* p = (float*)src.DataPointer;

        System.Text.StringBuilder sb = new();
        sb.AppendLine($"[Anima per-ch] {label} shape=[{batch},{channels},{h},{w}]:");
        for (int c = 0; c < channels; c++)
        {
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            double sum = 0, sumSq = 0, sumAbs = 0;
            long count = 0;
            for (int b = 0; b < batch; b++)
            {
                long bcOff = ((long)b * channels + c) * spatial;
                for (long i = 0; i < spatial; i++)
                {
                    float v = p[bcOff + i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                    sumSq += (double)v * v;
                    sumAbs += Math.Abs(v);
                    count++;
                }
            }
            double mean = count > 0 ? sum / count : 0;
            double std = count > 0 ? Math.Sqrt(Math.Max(0, sumSq / count - mean * mean)) : 0;
            double absMean = count > 0 ? sumAbs / count : 0;
            sb.AppendLine($"   ch{c,2}: min={min,7:F3} max={max,7:F3} mean={mean,7:F4} abs_mean={absMean,7:F4} std={std,7:F4}");
        }
        Logs.Info(sb.ToString().TrimEnd());

        if (!ReferenceEquals(src, t)) src.Dispose();
    }

    private static unsafe Tensor CloneF32(Tensor src)
    {
        Tensor copy = new Tensor(src.Shape, DType.F32);
        long bytes = src.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy(src.DataPointer, copy.DataPointer, bytes, bytes);
        return copy;
    }

    private static void ApplyStandardCfg(Tensor output, Tensor cond, Tensor uncond, float cfg)
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

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose() => Volatile.Write(ref _disposed, 1);
}
