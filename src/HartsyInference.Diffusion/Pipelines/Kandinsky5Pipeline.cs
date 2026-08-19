using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Kandinsky 5.0 text-to-image pipeline (kandinskylab/Kandinsky-5.0-T2I-Lite).
///
/// This pipeline only handles the diffusion side — the dual text encoder stack (Qwen2.5-VL +
/// CLIP-L) is not yet implemented as a HartsyInference text encoder. Callers must supply the
/// pre-computed Qwen sequence embeddings <c>[B, S, 3584]</c> and the CLIP pooled embeddings
/// <c>[B, 768]</c>, plus their negative-prompt counterparts. This matches the diffusers
/// <c>Kandinsky5T2IPipeline.__call__(prompt_embeds_qwen=, prompt_embeds_clip=, ...)</c> escape
/// hatch — production users can pre-compute embeddings once with a sidecar Python helper, and
/// once a Qwen2.5-VL text encoder lands in HartsyInference this pipeline can grow a tokenizer-fed
/// overload without touching the denoising path.</summary>
public sealed unsafe class Kandinsky5Pipeline : DiffusionPipelineBase
{
    private readonly Kandinsky5Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Kandinsky5Config _config;
    private readonly float _schedulerShift;
    private readonly float _vaeScalingFactor;
    private readonly float _vaeShiftFactor;

    /// <summary>Creates a new Kandinsky 5 pipeline. The VAE used for the Lite model is the Flux VAE
    /// (16-channel latent, 8× downsample), with shift/scale identical to <c>VaeConfig.Flux</c>. Img2img is
    /// unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="transformer">Kandinsky 5 transformer (use <see cref="Kandinsky5Config.Lite"/>).</param>
    /// <param name="vaeDecoder">Flux VAE decoder (16 channels).</param>
    /// <param name="config">Transformer configuration.</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. Diffusers config: <c>shift=5.0</c>.</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Flux: <c>0.3611</c>.</param>
    /// <param name="vaeShiftFactor">VAE shift factor. Flux: <c>0.1159</c>.</param>
    public Kandinsky5Pipeline(IBackend backend, Kandinsky5Transformer transformer, VaeDecoder vaeDecoder,
        Kandinsky5Config config, float schedulerShift = 5.0f,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : this(backend, transformer, vaeDecoder, vaeEncoder: null, config, schedulerShift, vaeScalingFactor, vaeShiftFactor)
    {
    }

    /// <summary>Creates a new Kandinsky 5 pipeline with both VAE halves loaded — required for img2img / inpaint
    /// (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromEmbeddings"/>). Configure the encoder
    /// with <see cref="VaeConfig.Flux"/> so its output is already normalized into the transformer's latent space
    /// (matching the <c>latent / scale + shift</c> un-normalization this pipeline applies before decode).</summary>
    public Kandinsky5Pipeline(IBackend backend, Kandinsky5Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, Kandinsky5Config config, float schedulerShift = 5.0f,
        float vaeScalingFactor = 0.3611f, float vaeShiftFactor = 0.1159f)
        : base(backend)
    {
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _schedulerShift = schedulerShift;
        _vaeScalingFactor = vaeScalingFactor;
        _vaeShiftFactor = vaeShiftFactor;
    }

    /// <summary>Generates an image from pre-computed text encoder outputs. Uses
    /// <c>FlowMatchEulerDiscreteScheduler</c> (shift defaults to 5.0 per the Lite scheduler config).
    /// An <see cref="ImageToImageRequest"/> selects img2img: the source is VAE-encoded (Flux VAE) and noised via
    /// flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a <see cref="VaeEncoder"/> on
    /// construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step latent blend + final
    /// pixel recomposite). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="qwenEmbeds">Qwen2.5-VL sequence embeddings <c>[B, S_t, in_text_dim]</c>.</param>
    /// <param name="clipPooled">CLIP-L pooled embeddings <c>[B, in_text_dim2]</c>.</param>
    /// <param name="negQwenEmbeds">Negative-prompt Qwen embeddings (only required if cfg &gt; 1).</param>
    /// <param name="negClipPooled">Negative-prompt CLIP pooled (only required if cfg &gt; 1).</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor qwenEmbeds, Tensor clipPooled,
        Tensor? negQwenEmbeds, Tensor? negClipPooled,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Kandinsky5Image.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && (negQwenEmbeds is null || negClipPooled is null))
            throw new ArgumentException(
                "cfgScale > 1.0 requires both negQwenEmbeds and negClipPooled.", nameof(negQwenEmbeds));

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
        Logs.Info($"Kandinsky5 {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Set up flow-match scheduler with shift=5.0 ──
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 2. Build initial latent [B, 16, latentH, latentW] (t2i: noise * initSigma; img2img: encode + AddNoise) ──
        TensorShape latentShape = new TensorShape(1, _config.InVisualDim, latentH, latentW);
        (Tensor latent, Tensor? sourceLatent) =
            BuildInitialLatent(request, scheduler, latentShape, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // ── 3. Denoising loop ──
        // Bulk-upload transformer weights before the denoise loop. Paired with FreeWeights
        // below the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Kandinsky5: starting denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor? uncond = null;
            Tensor? cond = null;
            try
            {
                if (useCfg)
                    uncond = _transformer.Forward(Backend, latent, t, negQwenEmbeds!, negClipPooled!);
                cond = _transformer.Forward(Backend, latent, t, qwenEmbeds, clipPooled);

                // Tensor statistics and raw dumps intentionally materialize device tensors. Keep those
                // readbacks behind the existing debug-dump switch; production sampling must not run global
                // host reductions or materialize device tensors merely to format informational log lines.
                if (Kandinsky5DebugDump.Enabled && (i == 0 || i == steps / 2 || i == steps - 1))
                {
                    Tensor diagnosticPred = useCfg
                        ? CfgHelper.ApplyCfg(uncond!, cond, cfgScale)
                        : cond;
                    try
                    {
                        Logs.Info($"[K5DIAG] step {i}: t={t:F2} sigma_idx noisePred.std={StdOf(diagnosticPred):F4} latent.std={StdOf(latent):F4}");
                        if (i == 0)
                        {
                            Kandinsky5DebugDump.Dump("step0_latent", latent);
                            Kandinsky5DebugDump.Dump("step0_velocity", diagnosticPred);
                        }
                    }
                    finally
                    {
                        if (!ReferenceEquals(diagnosticPred, cond)) diagnosticPred.Dispose();
                    }
                }

                Backend.CfgEulerStep(latent, cond, useCfg ? uncond! : cond,
                    useCfg ? cfgScale : 1.0f, scheduler.Dt(i));
            }
            finally
            {
                cond?.Dispose();
                uncond?.Dispose();
            }

            // Masked-inpaint blend: keep unmasked region on the source's flow-matching trajectory
            // by re-noising the source latent at the next step's sigma. Final step blends with the
            // clean source — no further denoising follows.
            if (latentMask is not null && sourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    noisedSource = new Tensor(latentShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourceLatent, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourceLatent;
                }
                MaskBlendUtilities.BlendChannelsInPlace(latent, noisedSource, latentMask);
                if (noisedSource != sourceLatent) noisedSource.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        sourceLatent?.Dispose();
        latentMask?.Dispose();

        if (Kandinsky5DebugDump.Enabled)
            Logs.Info($"[K5DIAG] final latent.std={StdOf(latent):F4}");
        Kandinsky5Transformer.DumpFinalLatent(latent);

        // ── 4. Free transformer weights before VAE decode (mirrors AuraFlow / Flux pattern). ──
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());

        // ── 5. Apply Flux VAE shift+scale before decoding: latent = latent / scale + shift ──
        Tensor decodeIn = new Tensor(latent.Shape, DType.F32);
        ApplyVaeShiftScale(decodeIn, latent, _vaeScalingFactor, _vaeShiftFactor);
        latent.Dispose();

        // ── 6. VAE decode (tiled to keep im2col workspace bounded). ──
        Logs.Verbose("Kandinsky5: decoding latents (tiled)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, decodeIn);
        decodeIn.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Kandinsky5 {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: VaeEncoder.Encode(source)
    /// (Flux-VAE-configured encoder → already in the transformer's normalized latent space) combined with fresh
    /// noise via flow-matching AddNoise at sigma[startStep].
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean source latent is returned
    /// alongside the noised latent for per-step blending. Caller disposes both. Source is null for txt2img and plain
    /// img2img.</para></summary>
    private (Tensor latent, Tensor? sourceLatent) BuildInitialLatent(TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler, TensorShape latentShape, int seed, int startStep,
        bool keepSourceLatent)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = TakeOrCreateNoise(request, latentShape, seed);
            Tensor latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            noise.Dispose();
            if (keepSourceLatent)
            {
                return (latent, sourceLatent);
            }
            sourceLatent.Dispose();
            return (latent, null);
        }

        Tensor t2iNoise = TakeOrCreateNoise(request, latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            Backend.Scale(scaled, t2iNoise, initSigma);
            t2iNoise.Dispose();
            return (scaled, null);
        }
        return (t2iNoise, null);
    }

    private static unsafe float StdOf(Tensor t)
    {
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        double mean = 0;
        for (long i = 0; i < n; i++) mean += p[i];
        mean /= n;
        double var = 0;
        for (long i = 0; i < n; i++) { double d = p[i] - mean; var += d * d; }
        return (float)Math.Sqrt(var / n);
    }

    /// <summary>Inverse of the VAE encoder's normalization: <c>x = x / scale + shift</c>. Flux/Kandinsky 5
    /// store latents in the shifted-scaled space, so we reverse it before decoding.</summary>
    private static void ApplyVaeShiftScale(Tensor output, Tensor input, float scale, float shift)
    {
        float* i = (float*)input.DataPointer;
        float* o = (float*)output.DataPointer;
        long n = input.ElementCount;
        float invScale = 1.0f / scale;
        for (long k = 0; k < n; k++)
            o[k] = i[k] * invScale + shift;
    }
}
