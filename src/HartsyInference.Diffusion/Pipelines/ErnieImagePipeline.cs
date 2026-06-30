using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>ERNIE-Image (Baidu, ~8B params, Apache-2.0) text-to-image pipeline. Orchestrates a custom Baidu text encoder → ERNIE-Image transformer (single-stream DiT with shared AdaLN) → Flux2-style 128-channel VAE → RGB output. Reference: <c>diffusers/pipelines/ernie_image/pipeline_ernie_image.py</c>.
///
/// Pipeline-level deltas vs other DiT pipelines:
/// <list type="bullet">
///   <item>**Text encoder swappable.** This pipeline accepts any <see cref="IErnieTextEncoder"/>; the published <c>baidu/ERNIE-Image</c> encoder is Mistral3-shaped and served by <see cref="ErnieImageLlamaTextEncoder"/>.</item>
///   <item>**Per-batch text length is tracked separately** and forwarded into the transformer so 3D RoPE can offset image positions by the actual non-padded text length.</item>
///   <item>**Patchify on top of the latent.** The Flux2-style VAE produces a 32-channel latent; the pipeline applies a 2×2 channel-fold (32 → 128) before feeding the transformer (mirrors <c>pipeline_ernie_image.py:_patchify_latents</c>) and undoes it before VAE decode.</item>
///   <item>**BatchNorm-style latent normalization.** The Flux2 VAE ships <c>bn.running_mean</c>/<c>bn.running_var</c>; the pipeline un-normalizes via these stats just before VAE decode.</item>
///   <item>**Standard CFG dual-pass.**</item>
/// </list>
///
/// The pipeline does NOT own the BN stats — it accepts an optional <c>vaeBnMean</c>/<c>vaeBnVar</c> pair via the constructor. If <c>null</c>, the latent is fed to the VAE without un-normalization (works for VAEs that don't ship BN-style stats).</summary>
public sealed unsafe class ErnieImagePipeline : DiffusionPipelineBase
{
    private readonly IErnieTextEncoder _textEncoder;
    private readonly ErnieImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly ErnieImageConfig _config;
    private readonly Tensor? _vaeBnMean;
    private readonly Tensor? _vaeBnVar;
    private readonly float _vaeBnEps;
    private readonly float _schedulerShift;

    /// <summary>Creates a new ERNIE-Image pipeline.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">A loaded <see cref="IErnieTextEncoder"/>. The diffusers reference uses <c>output.hidden_states[-2]</c> as conditioning — make sure your encoder honours that convention.</param>
    /// <param name="transformer">Pre-loaded ERNIE-Image transformer.</param>
    /// <param name="vaeDecoder">Pre-loaded Flux2-style VAE decoder (configure with <c>VaeConfig.Flux2</c>).</param>
    /// <param name="config">Configuration record (pinned to the loaded transformer).</param>
    /// <param name="vaeBnMean">Optional <c>[1, 32, 1, 1]</c> running-mean tensor for the Flux2 VAE BN-style un-normalization. Pass <c>null</c> to skip un-normalization.</param>
    /// <param name="vaeBnVar">Optional running-var tensor (same shape as <paramref name="vaeBnMean"/>).</param>
    /// <param name="vaeBnEps">Numerical epsilon used in <c>std = sqrt(var + eps)</c>. Default 1e-5 (matches diffusers' BN default).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. Default <b>4.0</b> per ERNIE-Image's
    /// <c>scheduler_config.json</c> (<c>shift=4.0</c>, static); ERNIE-Image-Turbo may differ.</param>
    public ErnieImagePipeline(IBackend backend, IErnieTextEncoder textEncoder, ErnieImageTransformer transformer,
        VaeDecoder vaeDecoder, ErnieImageConfig config,
        Tensor? vaeBnMean = null, Tensor? vaeBnVar = null, float vaeBnEps = 1e-5f,
        float schedulerShift = 4.0f)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
        // ApplyBnUnnormalize reads these as raw F32 (`(float*)DataPointer`), but the Flux2 VAE ships them as BF16.
        // Cast to F32 here so a BF16/F16 tensor isn't reinterpreted as F32 (which read garbage + ran off the end of
        // the 256-byte BF16 buffer → NaN in ~25 of 128 channels → flat-black output).
        _vaeBnMean = vaeBnMean is not null && vaeBnMean.DType != DType.F32 ? vaeBnMean.CastTo(DType.F32) : vaeBnMean;
        _vaeBnVar = vaeBnVar is not null && vaeBnVar.DType != DType.F32 ? vaeBnVar.CastTo(DType.F32) : vaeBnVar;
        _vaeBnEps = vaeBnEps;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized prompt + negative prompt token ids (use <c>ErnieTokenizer</c> from HartsyInference.Tokenizers: BOS-prefixed raw prompt, no padding). The token ids must already be padded (per-prompt) up to a single common <c>Tmax</c>; pass the corresponding real lengths in <paramref name="promptRealLen"/> and <paramref name="negativeRealLen"/>.</summary>
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
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.ErnieImage.Resolve(request);
        // ERNIE-Image: VAE is 8× spatial, then a 2×2 patchify in pipeline → effective 16× downscale.
        int effectiveDownscale = 16;
        if (width % effectiveDownscale != 0 || height % effectiveDownscale != 0)
            throw new ArgumentException($"Width and height must be divisible by {effectiveDownscale} for ERNIE-Image.");

        int latentH = height / effectiveDownscale;
        int latentW = width / effectiveDownscale;

        Logs.Info($"ERNIE-Image: Generating {width}x{height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts ─────────────────────────────────────────────
        bool useCfg = cfgScale > 1.0f;

        Logs.Info("Encoding prompt(s)...");
        // Bulk-upload text encoder weights once so its kernels don't pay per-op cache-miss
        // H2D transfers. Paired with FreeWeights below the VAE handoff. No-op on backends
        // without a weight cache.
        Backend.PreloadWeights(_textEncoder.EnumerateWeights());
        (Tensor condEmb, int[] condLens) = EncodeBatch(promptTokenIds, promptRealLen);
        Tensor? uncondEmb = null;
        int[]? uncondLens = null;
        if (useCfg)
        {
            (uncondEmb, uncondLens) = EncodeBatch(negativePromptTokenIds, negativeRealLen);
        }
        Logs.Info($"Text encoded in {sw.ElapsedMilliseconds}ms");
        ErnieDiag("condEmb", condEmb);
        if (uncondEmb is not null) ErnieDiag("uncondEmb", uncondEmb);

        // Free the text-encoder weights from VRAM now: they are only needed for the encode above, and
        // the ERNIE-Image TE (Ministral-3B, ~7.7 GB BF16) cannot coexist with the FP8 transformer
        // (~7.5 GB) on a 12 GB card. The encoded embeddings (condEmb/uncondEmb) are separate tensors and
        // remain valid. No-op on backends without a weight cache.
        Backend.Sync();
        Backend.FreeWeights(_textEncoder.EnumerateWeights());

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
            Backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 4. Denoising loop ─────────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights at
        // the VAE handoff (line ~160). No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting ERNIE-Image denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                Tensor uncondNoise = _transformer.Forward(Backend, latent, t, uncondEmb!, uncondLens!);
                Tensor condNoise = _transformer.Forward(Backend, latent, t, condEmb, condLens);
                noisePred = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
                uncondNoise.Dispose();
                condNoise.Dispose();
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condEmb, condLens);
            }

            if (i == 0) ErnieDiag("noisePred[0]", noisePred);
            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        ErnieDiag("finalLatent", latent);
        condEmb.Dispose();
        uncondEmb?.Dispose();

        // ── 5. Free transformer weights from VRAM before VAE decode (TE was already freed post-encode) ──
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // ── 6. BN-style un-normalization (Flux2 VAE ships running mean/var) ──
        if (_vaeBnMean is not null && _vaeBnVar is not null)
        {
            Tensor unnormed = ApplyBnUnnormalize(latent, _vaeBnMean, _vaeBnVar, _vaeBnEps);
            latent.Dispose();
            latent = unnormed;
            ErnieDiag("latent_postBN", latent);
        }

        // ── 7. Unpatchify [1, 128, latentH, latentW] → [1, 32, 2*latentH, 2*latentW] before VAE decode ──
        Tensor vaeIn = UnpatchifyLatent(latent);
        latent.Dispose();
        ErnieDiag("vaeIn", vaeIn);

        // ── 8. VAE decode ─────────────────────────────────────────────────
        // Full (non-tiled) decode: the tiled path produced horizontal BANDING — its non-stride-aligned last-tile
        // overlap (latent rows 0/56/64 at 1024) exceeds the fixed overlapLatent=8 tent blend, and per-tile GroupNorm
        // statistics differ across the seam → a ~50/50 average of two differently-normalized decodes. ERNIE's latent is
        // small ([1,32,≤128,≤128]) so a full decode fits the 4090 (the ✅ Flux.2 path also decodes full). Reverted to
        // DecodeTiled only if a future high-res case OOMs.
        Logs.Verbose("Decoding latents to image (full F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(Backend, vaeIn);
        vaeIn.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 9. RGB conversion ─────────────────────────────────────────────
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"ERNIE-Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary>Temporary diagnostic: logs mean/std/min/max/NaN of a tensor (forces D2H). Localizes the
    /// flat-black-output bug (conditioning vs velocity vs latent vs BN-unnorm vs VAE). Remove once ERNIE is verified.</summary>
    private static unsafe void ErnieDiag(string name, Tensor t)
    {
        if (Environment.GetEnvironmentVariable("ERNIE_DIAG") is null) return;  // env-gated: off in production
        ReadOnlySpan<float> s = t.AsReadOnlySpan<float>();
        double sum = 0, sum2 = 0; float min = float.MaxValue, max = float.MinValue; int nan = 0;
        for (int i = 0; i < s.Length; i++)
        {
            float v = s[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
            sum += v; sum2 += (double)v * v; if (v < min) min = v; if (v > max) max = v;
        }
        int n = s.Length - nan;
        double mean = n > 0 ? sum / n : 0;
        double std = n > 0 ? Math.Sqrt(Math.Max(0, sum2 / n - mean * mean)) : 0;
        HartsyInference.Core.Logging.Logs.Info($"[DIAG] {name} {t.Shape}: mean={mean:F4} std={std:F4} min={min:F4} max={max:F4} nan/inf={nan}/{s.Length}");
    }

    /// <summary>Runs the text encoder for a single (already padded) batch of token ids.</summary>
    private (Tensor Hidden, int[] Lens) EncodeBatch(int[] tokenIds, int realLen)
    {
        int[][] batch = [tokenIds];
        int[] lens = [realLen];
        return _textEncoder.Encode(Backend, batch, lens);
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
}
