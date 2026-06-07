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
public sealed unsafe class AnimaPipeline : DiffusionPipelineBase
{
    private readonly AnimaTransformer _transformer;
    private readonly AnimaLlmAdapter _llmAdapter;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly AnimaConfig _config;

    /// <summary>Creates an Anima t2i pipeline. Caller owns the components. The VAE is the
    /// <see cref="QwenImageVaeDecoder"/> (3D causal autoencoder collapsed to 2D for image mode);
    /// the standard <see cref="VaeDecoder"/> class can't load this checkpoint because the key
    /// layout and conv ranks differ. Img2img is unavailable; use the overload accepting a
    /// <see cref="QwenImageVaeEncoder"/> to enable it.</summary>
    public AnimaPipeline(IBackend backend, AnimaTransformer transformer, AnimaLlmAdapter llmAdapter,
        QwenImageVaeDecoder vaeDecoder, AnimaConfig config)
        : this(backend, transformer, llmAdapter, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates an Anima pipeline with both VAE halves loaded — required for img2img / inpaint
    /// (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromEmbeddings"/>).</summary>
    public AnimaPipeline(IBackend backend, AnimaTransformer transformer, AnimaLlmAdapter llmAdapter,
        QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder, AnimaConfig config)
        : base(backend)
    {
        _transformer = transformer;
        _llmAdapter = llmAdapter;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-computed Qwen-3 0.6B hidden states <c>[1, T, 1024]</c> + T5
    /// tokenizations of the same prompt(s). The LlmAdapter's main stream is built from
    /// <c>embed[t5TokenIds]</c>; the Qwen-3 states are the cross-attention K/V source. The adapter's output
    /// sequence length equals the T5 token count (NOT the Qwen-3 token count).
    /// <para>CFG dual-pass when <paramref name="cfgScale"/> &gt; 1 and both negative embeddings and negative
    /// T5 token ids are provided.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor textEmbeddings,
        int[] t5TokenIds,
        TextToImageRequest request,
        float cfgScale = 1.0f,
        Tensor? negativeTextEmbeddings = null,
        int[]? negativeT5TokenIds = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (cfgScale > 1.0f && (negativeTextEmbeddings is null || negativeT5TokenIds is null))
            throw new ArgumentException(
                "negativeTextEmbeddings and negativeT5TokenIds are required when cfgScale > 1.0.",
                nameof(negativeTextEmbeddings));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;

        // img2img / inpaint: validate source + compute the denoise start step + strength=0 short-circuit.
        Img2ImgSetup.Plan img2imgPlan = Img2ImgSetup.Prepare(request, height, width, steps);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException(
                "ImageToImageRequest requires a QwenImageVaeEncoder. Construct AnimaPipeline with the overload that accepts one.");
        if (img2imgPlan.PassThrough)
        {
            Logs.Info("Anima img2img strength=0 → returning source image unchanged.");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }

        Logs.Info($"Anima {(isImg2Img ? "img2img" : "t2i")}: {width}x{height}, {steps} steps, start={img2imgPlan.StartStep}, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        LogStats("textEmbeddings (Qwen3 0.6B output)", textEmbeddings);

        // Bulk-upload the LlmAdapter + transformer weights to the backend's weight cache so the
        // hundreds of per-op kernel launches inside Forward() don't each pay a cache-miss H2D
        // transfer. Backends without a weight cache (CPU, Vulkan) no-op. Paired with the
        // FreeWeights calls below the VAE handoff so VRAM cycles cleanly between generations.
        Backend.PreloadWeights(_llmAdapter.EnumerateWeights());
        Backend.PreloadWeights(_transformer.EnumerateWeights());

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
            // Refine through the LlmAdapter ONCE per generation (timestep-independent). Main stream =
            // embed[t5_ids], cross-attn K/V = Qwen-3 hidden states. Output seq_len = t5_ids.Length.
            refinedText = _llmAdapter.Forward(Backend, textEmbeddings, t5TokenIds);
        }
        LogStats("refinedText (LlmAdapter output)", refinedText);

        Tensor? refinedNegText = null;
        if (negativeTextEmbeddings is not null)
        {
            refinedNegText = bypassAdapter
                ? (negativeTextEmbeddings.DType == DType.F32 ? CloneF32(negativeTextEmbeddings) : negativeTextEmbeddings.CastTo(DType.F32))
                : _llmAdapter.Forward(Backend, negativeTextEmbeddings, negativeT5TokenIds!);
        }

        TensorShape latentShape = new(1, _config.InChannels, latentH, latentW);

        // Pad refinedText / refinedNegText to 512 tokens with zeros. Anima training (per
        // `tdrussell/diffusion-pipe/models/cosmos_predict2.py:_tokenize`) tokenizes T5 inputs to
        // `padding="max_length", max_length=512` and the upstream-blessed inference path
        // (`modelscope/DiffSynth-Studio/diffsynth/models/anima_dit.py:preprocess_text_embeds`) right-pads the
        // LlmAdapter output with zeros to exactly 512 tokens before feeding the DiT cross-attention. Without
        // padding, the cross-attention sees a much shorter sequence than at training time, the attention
        // softmax weights are concentrated on the wrong tokens, and the model produces noise.
        const int CrossAttnPaddedLen = 512;
        Tensor refinedTextPadded = PadSeq(refinedText, CrossAttnPaddedLen);
        refinedText.Dispose();
        refinedText = refinedTextPadded;
        if (refinedNegText is not null)
        {
            Tensor pn = PadSeq(refinedNegText, CrossAttnPaddedLen);
            refinedNegText.Dispose();
            refinedNegText = pn;
        }
        LogStats("refinedText (padded to 512)", refinedText);

        // Anima uses the Z-Image flow-matching schedule (per
        // `modelscope/DiffSynth-Studio/diffsynth/diffusion/flow_match.py`):
        //   sigmas = linspace(1, 0, N+1)[:-1]       # drop terminal 0
        //   sigmas = shift * sigmas / (1 + (shift-1) * sigmas)   # shift=3 for Z-Image / Anima
        // The network outputs velocity (v ≈ noise - x_clean); we integrate via plain Euler:
        //   x_{i+1} = x_i + v * (sigma_next - sigma)
        // No EDM preconditioning, no sigma_max=80 scaling. Initial latent is unit-variance Gaussian.
        const float Shift = 3.0f;
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            float lin = 1.0f - (float)i / steps;        // linspace(1, 0, N+1)[:-1]
            sigmas[i] = Shift * lin / (1.0f + (Shift - 1.0f) * lin);
        }
        sigmas[steps] = 0.0f;                            // terminal sigma
        Logs.Info($"Anima sigmas (Z-Image flow-match, shift={Shift}): [{sigmas[0]:F4}, {sigmas[1]:F4}, ..., {sigmas[^2]:F4}, {sigmas[^1]:F4}]");

        // Initial latent: fresh noise (t2i) or the VAE-encoded source noised to sigma[startStep] (img2img).
        int startStep = img2imgPlan.StartStep;
        Tensor latent;
        if (isImg2Img)
        {
            Tensor sourceLatent = _vaeEncoder!.Encode(Backend, ((ImageToImageRequest)request).SourceImage);  // [1, 16, latentH, latentW]
            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            latent = new Tensor(latentShape, DType.F32);
            AddNoiseFlowMatch(latent, sourceLatent, noise, sigmas[startStep]);   // x = (1-σ)·source + σ·noise
            sourceLatent.Dispose();
            noise.Dispose();
            Logs.Info($"Anima img2img: encoded source, AddNoise at sigma[{startStep}]={sigmas[startStep]:F4}.");
        }
        else
        {
            latent = SeedGenerator.CreateNoise(latentShape, seed);
        }
        LogStats("initial latent", latent);

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = sigmas[i];
            float sigmaNext = sigmas[i + 1];
            // Anima feeds `t = sigma` directly to the transformer (per anima-preview.py + anima_dit.py:
            // `timestep = timestep / 1000` then model receives the normalized t = sigma).
            float t = sigma;

            Tensor velocity = _transformer.Forward(Backend, latent, t, refinedText);
            if (cfgScale > 1.0f)
            {
                Tensor velUncond = _transformer.Forward(Backend, latent, t, refinedNegText!);
                // Standard SD3-form CFG: combined = uncond + scale * (cond - uncond).
                Tensor combined = CfgHelper.ApplyCfg(velUncond, velocity, cfgScale);
                velUncond.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            if (i == 0 || i == steps / 2 || i == steps - 1)
                LogStats($"velocity step={i} (t={t:F4}, sigma={sigma:F4}→{sigmaNext:F4})", velocity);

            Tensor newLatent = new(latentShape, DType.F32);
            FlowMatchStep(newLatent, latent, velocity, sigmaNext - sigma);
            velocity.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                // Attach the current latent so the backend's PreviewEncoder can render per-step previews
                // (without this, progress.Latent is null and TryEncode bails → no previews). Anima's latent
                // is [1, 16, H, W] NCHW (Qwen-Image VAE), decoded with Flux factors per LatentArchitecture.Anima.
                Latent = latent,
                LatentArch = LatentArchitecture.Anima,
            });
        }

        refinedText.Dispose();
        refinedNegText?.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        Backend.FreeWeights(_llmAdapter.EnumerateWeights());

        LogStats("final latent (pre-VAE)", latent);
        LogPerChannelStats("final latent (pre-VAE) per-channel", latent);

        Tensor decoded = _vaeDecoder.Decode(Backend, latent);
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

    /// <summary>Right-pads <paramref name="text"/> <c>[B, S, D]</c> to <paramref name="targetLen"/> tokens by
    /// appending all-zero rows. Caller owns the returned tensor and the input. Mirrors
    /// <c>diffsynth/models/anima_dit.py:preprocess_text_embeds</c>'s
    /// <c>F.pad(out, (0, 0, 0, 512 - out.shape[1]))</c>.</summary>
    private static unsafe Tensor PadSeq(Tensor text, int targetLen)
    {
        int batch = (int)text.Shape[0];
        int curLen = (int)text.Shape[1];
        int dim = (int)text.Shape[2];
        if (curLen == targetLen) return CloneF32(text);
        if (curLen > targetLen)
            throw new ArgumentException($"PadSeq: input seq_len {curLen} > target {targetLen}; truncation not supported.", nameof(text));

        TensorShape outShape = new TensorShape(batch, targetLen, dim);
        Tensor padded = new Tensor(outShape, DType.F32);
        float* srcPtr = (float*)text.DataPointer;
        float* dstPtr = (float*)padded.DataPointer;
        long copyBytes = (long)batch * curLen * dim * sizeof(float);
        // text is laid out [B, S, D] contiguously; the [0..curLen) prefix per batch followed by zeros.
        for (int b = 0; b < batch; b++)
        {
            float* src = srcPtr + (long)b * curLen * dim;
            float* dst = dstPtr + (long)b * targetLen * dim;
            Buffer.MemoryCopy(src, dst, curLen * dim * sizeof(float), curLen * dim * sizeof(float));
            // Remaining slots stay zero (default-initialised by Tensor allocator — verify in your runtime;
            // if it doesn't, add an explicit memset here). On SharpInference's tensor allocator, F32 tensors
            // are zero-initialised on alloc.
            float* zerosStart = dst + (long)curLen * dim;
            long zeroFloats = (long)(targetLen - curLen) * dim;
            for (long i = 0; i < zeroFloats; i++) zerosStart[i] = 0f;
        }
        return padded;
    }

    /// <summary>In-place scalar multiply: <c>t[i] *= s</c>. Used to scale the initial noise by sigma_max.</summary>
    private static unsafe void ScaleInPlace(Tensor t, float s)
    {
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= s;
    }

    /// <summary>Out-of-place scalar multiply: <c>out[i] = s * src[i]</c>. Used for the EDM
    /// <c>latent_model_input = latent * c_in</c> step.</summary>
    private static unsafe Tensor ScaleClone(Tensor src, float s)
    {
        Tensor result = new Tensor(src.Shape, DType.F32);
        float* sp = (float*)src.DataPointer;
        float* op = (float*)result.DataPointer;
        long n = src.Shape.ElementCount;
        for (long i = 0; i < n; i++) op[i] = sp[i] * s;
        return result;
    }

    /// <summary>Cosmos-style CFG: <c>out[i] = x0_cond[i] + scale * (x0_cond[i] - x0_uncond[i])</c>.
    /// NB: this differs from SD3/Flux's <c>uncond + scale * (cond - uncond)</c> — Cosmos uses cond as the
    /// baseline, not uncond, so the conditioning side is amplified additively. Matches
    /// diffusers' pipeline_cosmos2_text2image.py line 198.</summary>
    private static unsafe Tensor CosmosCfg(Tensor x0Cond, Tensor x0Uncond, float scale)
    {
        Tensor result = new Tensor(x0Cond.Shape, DType.F32);
        float* cp = (float*)x0Cond.DataPointer;
        float* up = (float*)x0Uncond.DataPointer;
        float* op = (float*)result.DataPointer;
        long n = x0Cond.Shape.ElementCount;
        for (long i = 0; i < n; i++) op[i] = cp[i] + scale * (cp[i] - up[i]);
        return result;
    }

    /// <summary>EDM denoised estimate: <c>x0[i] = cSkip * latent[i] + cOut * f[i]</c>.
    /// Matches diffusers' <c>noise_pred = c_skip * latents + c_out * F</c>.</summary>
    private static unsafe Tensor EdmDenoise(Tensor latent, Tensor f, float cSkip, float cOut)
    {
        Tensor x0 = new Tensor(latent.Shape, DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* fp = (float*)f.DataPointer;
        float* op = (float*)x0.DataPointer;
        long n = latent.Shape.ElementCount;
        for (long i = 0; i < n; i++) op[i] = cSkip * lp[i] + cOut * fp[i];
        return x0;
    }

    /// <summary>Convert EDM denoised estimate to FlowMatchEuler-compatible epsilon:
    /// <c>eps[i] = (latent[i] - x0[i]) / sigma</c>.</summary>
    private static unsafe Tensor EpsFromX0(Tensor latent, Tensor x0, float sigma)
    {
        Tensor eps = new Tensor(latent.Shape, DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* xp = (float*)x0.DataPointer;
        float* ep = (float*)eps.DataPointer;
        long n = latent.Shape.ElementCount;
        float invSigma = 1.0f / sigma;
        for (long i = 0; i < n; i++) ep[i] = (lp[i] - xp[i]) * invSigma;
        return eps;
    }

    /// <summary>FlowMatchEuler discrete step: <c>new_latent[i] = latent[i] + dt * eps[i]</c>.
    /// <paramref name="dt"/> is <c>sigma_next - sigma</c> (negative as sigmas decrease).</summary>
    private static unsafe void FlowMatchStep(Tensor newLatent, Tensor latent, Tensor eps, float dt)
    {
        float* np = (float*)newLatent.DataPointer;
        float* lp = (float*)latent.DataPointer;
        float* ep = (float*)eps.DataPointer;
        long n = latent.Shape.ElementCount;
        for (long i = 0; i < n; i++) np[i] = lp[i] + dt * ep[i];
    }

    /// <summary>Flow-matching forward noising for img2img: <c>out = (1 − sigma)·source + sigma·noise</c>.
    /// Mirrors the Z-Image / Anima schedule where the clean latent sits at sigma=0 and pure noise at the
    /// initial sigma, so the denoise loop resumed at step <c>startStep</c> integrates back to the source.</summary>
    private static unsafe void AddNoiseFlowMatch(Tensor outLatent, Tensor source, Tensor noise, float sigma)
    {
        float* op = (float*)outLatent.DataPointer;
        float* sp = (float*)source.DataPointer;
        float* xp = (float*)noise.DataPointer;
        long n = outLatent.Shape.ElementCount;
        float oneMinus = 1.0f - sigma;
        for (long i = 0; i < n; i++) op[i] = oneMinus * sp[i] + sigma * xp[i];
    }
}
