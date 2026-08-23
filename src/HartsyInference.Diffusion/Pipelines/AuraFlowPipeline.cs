using HartsyInference.Diffusion.Sampling;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>AuraFlow text-to-image pipeline (<c>AuraFlowPipeline</c>). Encodes prompts with Pile-T5-XL (joint_attention_dim=2048), runs the AuraFlow MMDiT transformer with classifier-free guidance, then decodes with the SDXL-compatible 4-channel VAE. Uses a static-shift FlowMatch scheduler. Mirrors the diffusers pipeline at <c>diffusers/pipelines/aura_flow/pipeline_aura_flow.py</c>.</summary>
public sealed class AuraFlowPipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly AuraFlowTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly AuraFlowConfig _config;
    private readonly float _schedulerShift;

    /// <summary>Standard-profile residency (HARTSY_KEEP_MODELS): transformer weights stay GPU-resident across generations. On a prompt-cache miss the Pile-T5-XL encoder is preloaded/freed around the encode as before.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache: repeat prompts skip the whole Pile-T5-XL phase. Keyed on token ids +
    // attention masks; tensors are host-materialized at store time (PrependRegisterTokens and the
    // batched-context build read them on the host anyway).
    private int[]? _teKeyCond, _teKeyUncond, _teKeyCondMask, _teKeyUncondMask;
    private Tensor? _cachedCond;
    private Tensor? _cachedUncond;

    /// <summary>Creates a new AuraFlow pipeline with all components pre-loaded. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">Pile-T5-XL text encoder (output dim 2048, max length 256).</param>
    /// <param name="transformer">AuraFlow MMDiT transformer.</param>
    /// <param name="vaeDecoder">SDXL VAE decoder (<see cref="VaeConfig.Sdxl"/>; 4-channel latent, scaling=0.13025).</param>
    /// <param name="config">AuraFlow configuration (use <see cref="AuraFlowConfig.V03"/> for fal/AuraFlow-v0.3).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. AuraFlow uses a static shift; default 1.73 matches the diffusers config.</param>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, AuraFlowConfig config, float schedulerShift = 1.73f)
        : this(backend, t5, transformer, vaeDecoder, vaeEncoder: null, config, schedulerShift)
    {
    }

    /// <summary>Creates a new AuraFlow pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Configure the encoder with <see cref="VaeConfig.AuraFlow"/> so its output is already in the transformer's latent space.</summary>
    public AuraFlowPipeline(IBackend backend, T5TextEncoder t5, AuraFlowTransformer transformer,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, AuraFlowConfig config, float schedulerShift = 1.73f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized Pile-T5-XL input with optional CFG. An <see cref="ImageToImageRequest"/> selects img2img: the source is VAE-encoded and noised via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a <see cref="VaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step latent blend + final pixel recomposite). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from Pile-T5-XL tokenizer (max length 256).</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs (same length as <paramref name="promptTokenIdsT5"/>).</param>
    /// <param name="promptAttentionMaskT5">Optional T5 attention mask for the prompt (1=attend, 0=pad).</param>
    /// <param name="negativeAttentionMaskT5">Optional T5 attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
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
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.AuraFlow.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;

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
        Logs.Info($"AuraFlow {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with Pile-T5-XL (with a cross-generation prompt-embedding cache) ──
        bool useCfg = cfgScale > 1.0f;
        bool teCacheHit = _cachedCond is not null
            && TokensEqual(_teKeyCond, promptTokenIdsT5) && TokensEqual(_teKeyCondMask, promptAttentionMaskT5)
            && (!useCfg || (_cachedUncond is not null
                && TokensEqual(_teKeyUncond, negativePromptTokenIdsT5) && TokensEqual(_teKeyUncondMask, negativeAttentionMaskT5)));

        Tensor condContext;
        Tensor? uncondContext;
        if (teCacheHit)
        {
            condContext = _cachedCond!;
            uncondContext = useCfg ? _cachedUncond : null;
            Logs.Info("AuraFlow prompt-embedding cache hit — text encoding skipped");
        }
        else
        {
            Logs.Info("Encoding text with Pile-T5-XL...");
            // Bulk-upload T5 weights once so the encoder's many kernel launches don't each pay a
            // per-op cache-miss H2D transfer; freed right after the encode (the cache makes repeat
            // prompts skip this phase entirely).
            Backend.PreloadWeights(_t5.EnumerateWeights());

            int[][] batchT5 = [promptTokenIdsT5];
            int[][]? batchMask = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
            condContext = _t5.Encode(Backend, batchT5, batchMask);

            uncondContext = null;
            if (useCfg)
            {
                int[][] negBatchT5 = [negativePromptTokenIdsT5];
                int[][]? negBatchMask = negativeAttentionMaskT5 is not null ? [negativeAttentionMaskT5] : null;
                uncondContext = _t5.Encode(Backend, negBatchT5, negBatchMask);
            }
            Backend.FreeWeights(_t5.EnumerateWeights());
            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

            // Host-materialize (the T5 outputs are live GPU activations until touched) and cache.
            unsafe
            {
                _ = (nint)condContext.DataPointer;
                if (uncondContext is not null) _ = (nint)uncondContext.DataPointer;
            }
            _cachedCond?.Dispose();
            _cachedUncond?.Dispose();
            _cachedCond = condContext;
            _cachedUncond = uncondContext;
            _teKeyCond = (int[])promptTokenIdsT5.Clone();
            _teKeyCondMask = (int[]?)promptAttentionMaskT5?.Clone();
            _teKeyUncond = useCfg ? (int[])negativePromptTokenIdsT5.Clone() : null;
            _teKeyUncondMask = (int[]?)negativeAttentionMaskT5?.Clone();
        }

        // ── 2. Set up flow-match scheduler (static shift) ────────────────
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial latent [1, 4, latentH, latentW] (t2i: noise * initSigma; img2img: encode + AddNoise at sigma[startStep]) ──
        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        (Tensor latent, Tensor? sourceLatent) =
            BuildInitialLatent(request, scheduler, latentShape, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // ── 4. Denoising loop ────────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop (touched every step, every
        // block). Under HARTSY_KEEP_MODELS they stay resident across generations.
        Stopwatch preloadSw = Stopwatch.StartNew();
        if (!_ditResident)
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
        }
        Logs.Verbose($"[auraflow-phase] DiT preload {(_ditResident ? "0 (resident)" : preloadSw.ElapsedMilliseconds.ToString())}ms");

        Logs.Info("Starting AuraFlow denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        Stopwatch denoiseSw = Stopwatch.StartNew();

        // Drain-free packed path (plain t2i/img2img): the latent lives in patch-token space on the
        // device for the whole loop — patchify once, dual-pass ForwardTokens per step, in-place
        // device CfgEulerStep (flow-match Euler: x += v·(σ[i+1]−σ[i])), unpatchify once at the end.
        // Masked inpaint keeps the reference host loop (per-step blend needs the spatial latent).
        // HARTSY_AURAFLOW_PACKED=0 is the kill-switch (A/B against the reference loop).
        bool fusedLoop = !isMaskedInpaint && EnvSwitch.IsEnabled("HARTSY_AURAFLOW_PACKED", defaultOn: true);

        // Sampler selection (2026-08-20). AuraFlow is flow-matching, so it had no user-selectable sampler at all
        // before this. The sampler owns the integrator and the sigma spacing; the family keeps owning its own sigma
        // range (static shift), and the dual-pass CFG stays inside the predictor closure below.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "AuraFlow",
            startsFromNoisedInit: startStep > 0);
        bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);

        // The host/reference loop below does not consult the sampler at all, so a non-default selection there would
        // be silently dropped — the exact failure this whole change removes. Refuse by name instead. Note this also
        // covers HARTSY_AURAFLOW_PACKED=0, the packed-path kill-switch: an A/B run against the reference loop cannot
        // honour a sampler either.
        if (!fusedLoop && nonDefaultSampler)
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' runs only on AuraFlow's packed drain-free path, and this "
                + "generation fell back to the reference loop (masked inpaint, or HARTSY_AURAFLOW_PACKED=0). Drop the "
                + "sampler selection, or drop the feature that forced the fallback.");
        }

        // The predictor closure needs the timestep table, and `timesteps` is a ReadOnlySpan (a ref local) that a
        // lambda cannot capture. One small array copy per generation.
        float[] timestepTable = timesteps.ToArray();
        if (fusedLoop)
        {
            int gridH = latentH / _config.PatchSize;
            int gridW = latentW / _config.PatchSize;
            Tensor latentTokens = _transformer.PatchifyToTokens(latent);
            latent.Dispose();

            for (int i = startStep; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                float t = timesteps[i];

                // AuraFlow does NOT pre-combine: the fused device op applies uncond-anchored CFG at the real scale,
                // so the predictor hands back the raw pair and reports cfgScale rather than 1.0.
                DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                    PredictionType.FlowVelocity,
                    (x, s, stepIndex) =>
                    {
                        // The transformer is conditioned on the 0-1000 timestep scale, which for a flow-match family
                        // IS sigma·1000. On-schedule sigmas reuse the loop's own `timesteps[i]` entry rather than
                        // recomputing `s·1000`: the two are equal mathematically, but the F32 round trip through
                        // x1000 is not exact (1000 is not a power of two), so substituting one for the other would
                        // shift every existing AuraFlow generation by an ulp of conditioning. Only a genuine
                        // off-schedule sub-step — what a second-order sampler evaluates at — takes the scaled value;
                        // there is no precomputed timestep for it.
                        float stepT = stepIndex < steps && s == scheduler.SigmaAt(stepIndex)
                            ? timestepTable[stepIndex]
                            : s * 1000.0f;
                        Tensor cond = _transformer.ForwardTokens(Backend, x, stepT, condContext, gridH, gridW);
                        if (!useCfg)
                        {
                            return new DenoisePrediction(cond, cond);
                        }
                        // The transformer's two-slot text-token cache holds both branches' conditioning,
                        // so each is projected once per generation, not once per step.
                        Tensor uncond = _transformer.ForwardTokens(Backend, x, stepT, uncondContext!, gridH, gridW);
                        return new DenoisePrediction(cond, uncond, cfgScale);
                    });
                if (i == startStep)
                {
                    sampler.Reset(latentTokens.Shape);
                }
                sampler.Step(Backend, latentTokens, predictor, i);

                stepSw.Stop();
                Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
                bool emitPreview = onProgress is not null && ((i - startStep) % 4 == 3 || i == steps - 1);
                if (emitPreview)
                {
                    Tensor previewLatent = new Tensor(latentShape, DType.F32);
                    Backend.UnpatchifyTokens(previewLatent, latentTokens, _config.OutChannels, gridH, gridW, _config.PatchSize, innerChannelFastest: false);
                    try
                    {
                        onProgress!.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                        {
                            Latent = previewLatent,
                            LatentArch = LatentArchitecture.AuraFlow,
                        });
                    }
                    finally { previewLatent.Dispose(); }
                }
                else
                {
                    onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
                }
            }

            latent = new Tensor(latentShape, DType.F32);
            Backend.UnpatchifyTokens(latent, latentTokens, _config.OutChannels, gridH, gridW, _config.PatchSize, innerChannelFastest: false);
            latentTokens.Dispose();
            // Host-materialize for the tiled-VAE fallback's host slicing (and any host consumer).
            Backend.Sync();
            unsafe { _ = (nint)latent.DataPointer; }
        }
        else
        {
            for (int i = startStep; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                float t = timesteps[i];

                Tensor noisePred;
                if (useCfg)
                {
                    Tensor uncondNoise = _transformer.Forward(Backend, latent, t, uncondContext!);
                    Tensor condNoise = _transformer.Forward(Backend, latent, t, condContext);
                    noisePred = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
                    uncondNoise.Dispose();
                    condNoise.Dispose();
                }
                else
                {
                    noisePred = _transformer.Forward(Backend, latent, t, condContext);
                }

                Tensor newLatent = new Tensor(latentShape, DType.F32);
                scheduler.Step(newLatent, noisePred, latent, i);
                noisePred.Dispose();
                latent.Dispose();
                latent = newLatent;

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
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                {
                    Latent = latent,
                    LatentArch = LatentArchitecture.AuraFlow,
                });
            }
        }
        Logs.Verbose($"[auraflow-phase] denoise {denoiseSw.ElapsedMilliseconds}ms ({(fusedLoop ? "packed" : "host")} loop)");

        if (!ReferenceEquals(condContext, _cachedCond)) condContext.Dispose();
        if (uncondContext is not null && !ReferenceEquals(uncondContext, _cachedUncond)) uncondContext.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        AuraFlowTransformer.DumpFinalLatent(latent);

        // Free transformer weights before VAE decode unless resident (the banded-conv VAE fits
        // beside the DiT on 24 GB; smaller cards keep the legacy evict via HARTSY_KEEP_MODELS=0).
        Backend.Sync();
        if (KeepModelsResident)
        {
            _ditResident = true;
        }
        else
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
        }

        // ── 5. VAE decode ────────────────────────────────────────────────
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
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
        Logs.Info($"AuraFlow {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Compares token/mask arrays for the prompt-embedding cache key (null-tolerant: two nulls match).</summary>
    private static bool TokensEqual(int[]? cached, int[]? incoming)
        => cached is null ? incoming is null : incoming is not null && cached.AsSpan().SequenceEqual(incoming);

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: VaeEncoder.Encode(source) (SDXL-family 4-channel VAE, scaling applied inside the encoder) combined with fresh noise via flow-matching AddNoise at sigma[startStep].
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
}
