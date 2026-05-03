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

/// <summary>ERNIE-Image (Baidu, ~8B params, Apache-2.0) text-to-image pipeline. Orchestrates a custom Baidu text encoder → ERNIE-Image transformer (single-stream DiT with shared AdaLN) → Flux2-style 128-channel VAE → RGB output. Reference: <c>diffusers/pipelines/ernie_image/pipeline_ernie_image.py</c>.
///
/// Pipeline-level deltas vs other DiT pipelines:
/// <list type="bullet">
///   <item>**Text encoder swappable.** ERNIE's encoder architecture is determined at runtime from the published HF repo's <c>text_encoder/config.json</c>; this pipeline accepts any <see cref="IErnieTextEncoder"/>. See <see cref="ErnieImagePlaceholderTextEncoder"/> (compile-time stub) and <see cref="ErnieImageLlamaTextEncoder"/> (Llama-shaped fallback).</item>
///   <item>**Per-batch text length is tracked separately** and forwarded into the transformer so 3D RoPE can offset image positions by the actual non-padded text length.</item>
///   <item>**Patchify on top of the latent.** The Flux2-style VAE produces a 32-channel latent; the pipeline applies a 2×2 channel-fold (32 → 128) before feeding the transformer (mirrors <c>pipeline_ernie_image.py:_patchify_latents</c>) and undoes it before VAE decode.</item>
///   <item>**BatchNorm-style latent normalization.** The Flux2 VAE ships <c>bn.running_mean</c>/<c>bn.running_var</c>; the pipeline un-normalizes via these stats just before VAE decode.</item>
///   <item>**Standard CFG dual-pass.**</item>
/// </list>
///
/// The pipeline does NOT own the BN stats — it accepts an optional <c>vaeBnMean</c>/<c>vaeBnVar</c> pair via the constructor. If <c>null</c>, the latent is fed to the VAE without un-normalization (works for VAEs that don't ship BN-style stats).</summary>
public sealed unsafe class ErnieImagePipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly IErnieTextEncoder _textEncoder;
    private readonly ErnieImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly ErnieImageConfig _config;
    private readonly Tensor? _vaeBnMean;
    private readonly Tensor? _vaeBnVar;
    private readonly float _vaeBnEps;
    private readonly float _schedulerShift;
    private int _disposed;

    /// <summary>Creates a new ERNIE-Image pipeline.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">A loaded <see cref="IErnieTextEncoder"/>. The diffusers reference uses <c>output.hidden_states[-2]</c> as conditioning — make sure your encoder honours that convention.</param>
    /// <param name="transformer">Pre-loaded ERNIE-Image transformer.</param>
    /// <param name="vaeDecoder">Pre-loaded Flux2-style VAE decoder (configure with <c>VaeConfig.Flux2</c>).</param>
    /// <param name="config">Configuration record (pinned to the loaded transformer).</param>
    /// <param name="vaeBnMean">Optional <c>[1, 32, 1, 1]</c> running-mean tensor for the Flux2 VAE BN-style un-normalization. Pass <c>null</c> to skip un-normalization.</param>
    /// <param name="vaeBnVar">Optional running-var tensor (same shape as <paramref name="vaeBnMean"/>).</param>
    /// <param name="vaeBnEps">Numerical epsilon used in <c>std = sqrt(var + eps)</c>. Default 1e-5 (matches diffusers' BN default).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. Default 1.0 (no shift) — set per the model's <c>scheduler_config.json</c>.</param>
    public ErnieImagePipeline(IBackend backend, IErnieTextEncoder textEncoder, ErnieImageTransformer transformer,
        VaeDecoder vaeDecoder, ErnieImageConfig config,
        Tensor? vaeBnMean = null, Tensor? vaeBnVar = null, float vaeBnEps = 1e-5f,
        float schedulerShift = 1.0f)
    {
        _backend = backend;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        _vaeBnMean = vaeBnMean;
        _vaeBnVar = vaeBnVar;
        _vaeBnEps = vaeBnEps;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized prompt + negative prompt token ids. The token ids must already be padded (per-prompt) up to a single common <c>Tmax</c>; pass the corresponding real lengths in <paramref name="promptRealLen"/> and <paramref name="negativeRealLen"/>. Use the placeholder encoder if you only need to test wiring without real text conditioning.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativePromptTokenIds,
        int promptRealLen,
        int negativeRealLen,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        // ERNIE-Image: VAE is 8× spatial, then a 2×2 patchify in pipeline → effective 16× downscale.
        int effectiveDownscale = 16;
        if (request.Width % effectiveDownscale != 0 || request.Height % effectiveDownscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {effectiveDownscale} for ERNIE-Image.");

        int latentH = request.Height / effectiveDownscale;
        int latentW = request.Width / effectiveDownscale;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"ERNIE-Image: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts ─────────────────────────────────────────────
        bool useCfg = cfgScale > 1.0f;

        Logs.Info("Encoding prompt(s)...");
        (Tensor condEmb, int[] condLens) = EncodeBatch(promptTokenIds, promptRealLen);
        Tensor? uncondEmb = null;
        int[]? uncondLens = null;
        if (useCfg)
        {
            (uncondEmb, uncondLens) = EncodeBatch(negativePromptTokenIds, negativeRealLen);
        }
        Logs.Info($"Text encoded in {sw.ElapsedMilliseconds}ms");

        // ── 2. Initial noise: [1, 128, latentH, latentW]. The transformer expects 128-channel latents already. ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 3. Flow-match Euler scheduler ─────────────────────────────────
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

        // ── 4. Denoising loop ─────────────────────────────────────────────
        Logs.Info("Starting ERNIE-Image denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor uncondNoise = _transformer.Forward(_backend, latent, t, uncondEmb!, uncondLens!);
                Tensor condNoise = _transformer.Forward(_backend, latent, t, condEmb, condLens);
                noisePred = ApplyCfg(uncondNoise, condNoise, cfgScale);
                uncondNoise.Dispose();
                condNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(_backend, latent, t, condEmb, condLens);
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condEmb.Dispose();
        uncondEmb?.Dispose();

        // ── 5. Free transformer + text encoder weights from VRAM before VAE decode ──
        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());
        _backend.FreeWeights(_textEncoder.EnumerateWeights());

        // ── 6. BN-style un-normalization (Flux2 VAE ships running mean/var) ──
        if (_vaeBnMean is not null && _vaeBnVar is not null)
        {
            Tensor unnormed = ApplyBnUnnormalize(latent, _vaeBnMean, _vaeBnVar, _vaeBnEps);
            latent.Dispose();
            latent = unnormed;
        }

        // ── 7. Unpatchify [1, 128, latentH, latentW] → [1, 32, 2*latentH, 2*latentW] before VAE decode ──
        Tensor vaeIn = UnpatchifyLatent(latent);
        latent.Dispose();

        // ── 8. VAE decode ─────────────────────────────────────────────────
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, vaeIn);
        vaeIn.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 9. RGB conversion ─────────────────────────────────────────────
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"ERNIE-Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, request.Width, request.Height, seed);
    }

    /// <summary>Runs the text encoder for a single (already padded) batch of token ids.</summary>
    private (Tensor Hidden, int[] Lens) EncodeBatch(int[] tokenIds, int realLen)
    {
        int[][] batch = [tokenIds];
        int[] lens = [realLen];
        return _textEncoder.Encode(_backend, batch, lens);
    }

    /// <summary>CFG combine: <c>out = uncond + scale * (cond - uncond)</c>.</summary>
    private static Tensor ApplyCfg(Tensor uncond, Tensor cond, float scale)
    {
        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uPtr = (float*)uncond.DataPointer;
        float* cPtr = (float*)cond.DataPointer;
        float* oPtr = (float*)output.DataPointer;
        long count = uncond.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            oPtr[i] = uPtr[i] + scale * (cPtr[i] - uPtr[i]);
        return output;
    }

    /// <summary>Reverses the pipeline-level 2×2 channel-fold: <c>[1, 128, H, W] → [1, 32, 2H, 2W]</c>. Mirrors <c>pipeline_ernie_image.py:_unpatchify_latents</c>.</summary>
    private static Tensor UnpatchifyLatent(Tensor packed)
    {
        int batch = (int)packed.Shape[0];
        int packedC = (int)packed.Shape[1];
        int packedH = (int)packed.Shape[2];
        int packedW = (int)packed.Shape[3];
        if (packedC % 4 != 0)
            throw new ArgumentException($"Latent channels {packedC} must be divisible by 4.", nameof(packed));

        int outC = packedC / 4;
        int outH = packedH * 2;
        int outW = packedW * 2;

        TensorShape shape = new TensorShape(batch, outC, outH, outW);
        Tensor output = new Tensor(shape, packed.DType);

        // Diffusers Python (matched verbatim):
        //   l.reshape(b, c//4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c//4, h*2, w*2)
        // Per-element index: source[b, oc*4 + (ph*2 + pw), ph_h?, ph_w?] is wrong — let me derive.
        // After reshape(b, c//4, 2, 2, h, w): axes are (b, oc, ph, pw, h, w).
        // After permute(0, 1, 4, 2, 5, 3): axes become (b, oc, h, ph, w, pw).
        // Final reshape merges (h, ph) → 2h and (w, pw) → 2w.
        // So output[b, oc, dst_h, dst_w] where dst_h = h*2 + ph, dst_w = w*2 + pw maps from
        // src[b, oc, ph, pw, h, w] = packed[b, oc*4 + ph*2 + pw, h, w].
        float* srcPtr = (float*)packed.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outC; oc++)
            {
                for (int h = 0; h < packedH; h++)
                {
                    for (int w = 0; w < packedW; w++)
                    {
                        for (int ph = 0; ph < 2; ph++)
                        {
                            for (int pw = 0; pw < 2; pw++)
                            {
                                int srcChannel = oc * 4 + ph * 2 + pw;
                                long srcOff = (((long)b * packedC + srcChannel) * packedH + h) * packedW + w;
                                int dstH = h * 2 + ph;
                                int dstW = w * 2 + pw;
                                long dstOff = (((long)b * outC + oc) * outH + dstH) * outW + dstW;
                                dstPtr[dstOff] = srcPtr[srcOff];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Un-applies the Flux2 VAE's BN-style normalization in-place: <c>z = z * sqrt(var + eps) + mean</c>. Both <paramref name="bnMean"/> and <paramref name="bnVar"/> are <c>[1, C, 1, 1]</c> (or <c>[C]</c> — we read them as flat <c>C</c>).</summary>
    private static Tensor ApplyBnUnnormalize(Tensor latent, Tensor bnMean, Tensor bnVar, float eps)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        long meanCount = bnMean.Shape.ElementCount;
        long varCount = bnVar.Shape.ElementCount;
        if (meanCount != channels || varCount != channels)
            throw new ArgumentException(
                $"BN mean/var element counts ({meanCount}/{varCount}) must equal latent channels ({channels}).");

        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* meanPtr = (float*)bnMean.DataPointer;
        float* varPtr = (float*)bnVar.DataPointer;

        long spatial = (long)height * width;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float std = MathF.Sqrt(varPtr[c] + eps);
                float mean = meanPtr[c];
                long base_ = ((long)b * channels + c) * spatial;
                for (long i = 0; i < spatial; i++)
                    outPtr[base_ + i] = inPtr[base_ + i] * std + mean;
            }
        }
        return output;
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Disposes the pipeline. Does NOT dispose the backend / encoder / transformer / VAE — those are shared resources owned by the caller.</summary>
    public void Dispose() => Volatile.Write(ref _disposed, 1);
}
