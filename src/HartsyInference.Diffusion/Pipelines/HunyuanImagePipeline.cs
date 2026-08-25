using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Hunyuan Image 2.1 text-to-image pipeline. The transformer takes per-token text embeddings at 3584-dim (Qwen2.5-VL-7B hidden) plus an optional 1472-dim secondary stream (ByT5 in the upstream model). Conditioning is driven by the timestep + optional distilled-guidance embedding — there is no pooled-conditioning path, so the CLIP encoder constructor parameter is accepted for API compatibility but unused. The pipeline patchifies the latent before the transformer call (<c>patch_size</c> from config) and unpatchifies after, then VAE-decodes through the 32-channel Hunyuan VAE.
/// <para>Two text-encode paths: the primary <see cref="HunyuanImageQwenTextEncoder"/> path matching the
/// upstream model (template → pad 1034 → <c>hidden_states[-3]</c> → drop the 34-token prefix), and a
/// legacy CLIP-L + T5-XXL constructor kept for repackaged checkpoints that bundle those encoders.
/// The ByT5 glyph branch (1472-dim secondary stream for quoted text) is not implemented yet — the
/// transformer receives <c>encoderHidden2 = null</c>, equivalent to diffusers' zero-tensor fallback for
/// prompts without quoted text. TODO (validation-gated): add a ByT5-small encoder for glyph prompts.</para></summary>
public sealed unsafe class HunyuanImagePipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder? _clipEncoder;
    private readonly T5TextEncoder? _t5Encoder;
    private readonly HunyuanImageQwenTextEncoder? _qwenEncoder;
    private readonly HunyuanImageTransformer _transformer;
    /// <summary>Optional encoder half (configure with <c>VaeConfig.HunyuanImage</c> — its 6-stage block list gives the 32x downscale this family needs). Required for img2img; an <see cref="ImageToImageRequest"/> without it throws. Set through an initializer rather than a fourth constructor overload, matching how the backends are supplied.</summary>
    public VaeEncoder? VaeEncoder { get; init; }

    /// <summary>The bespoke 2.1 pixel-shuffle encoder — the img2img/inpaint unlock. Takes priority over <see cref="VaeEncoder"/> (which the generic loader cannot build for this VAE anyway).</summary>
    public HunyuanImageVaeEncoder? HyVaeEncoder { get; init; }

    private readonly VaeDecoder _vaeDecoder;
    private readonly HunyuanImageVaeDecoder? _hyVaeDecoder;
    private readonly HunyuanImageConfig _config;

    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop FreeWeights + next-gen re-upload). The 7B Qwen2.5-VL TE cannot coexist with the resident DiT on 24 GB, so a prompt-cache MISS under this flag frees the DiT first, encodes, then re-preloads — repeat prompts skip both (the QwenImagePipeline staging pattern). Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables).</summary>
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used), keyed on the Qwen token ids — the
    // FLite/QwenImage pattern. A hit skips the whole TE phase (DiT evict + TE preload + encode + free).
    // NOTE: when the ByT5 glyph stream gets wired (encoderHidden2), its embedding must join this cache
    // under the same key — both streams derive from the same prompt.
    private int[]? _cachedCondKey;
    private Tensor? _cachedCond;
    private int[]? _cachedUncondKey;
    private Tensor? _cachedUncond;

    /// <summary>Creates a Hunyuan Image pipeline with the primary Qwen2.5-VL-7B text encoder (matches <c>tencent/HunyuanImage-2.1</c>).</summary>
    public HunyuanImagePipeline(IBackend backend, HunyuanImageQwenTextEncoder qwenEncoder,
        HunyuanImageTransformer transformer, VaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _qwenEncoder = qwenEncoder ?? throw new ArgumentNullException(nameof(qwenEncoder));
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Creates a pipeline with the faithful HunyuanImage pixel-shuffle VAE decoder (the generic <see cref="VaeDecoder"/> cannot express its conv→depth-to-space upsamplers).</summary>
    public HunyuanImagePipeline(IBackend backend, HunyuanImageQwenTextEncoder qwenEncoder,
        HunyuanImageTransformer transformer, HunyuanImageVaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _qwenEncoder = qwenEncoder ?? throw new ArgumentNullException(nameof(qwenEncoder));
        _transformer = transformer;
        _hyVaeDecoder = vaeDecoder;
        _vaeDecoder = null!;
        _config = config;
    }

    /// <summary>Legacy/fallback constructor using CLIP-L + T5-XXL (repackaged checkpoints). <paramref name="clipEncoder"/> is accepted for API compatibility but not consumed — Hunyuan's pooled-conditioning path is replaced by the timestep + distilled-guidance embedding inside the transformer.</summary>
    public HunyuanImagePipeline(IBackend backend, ClipTextEncoder? clipEncoder, T5TextEncoder t5Encoder,
        HunyuanImageTransformer transformer, VaeDecoder vaeDecoder, HunyuanImageConfig config)
        : base(backend)
    {
        _clipEncoder = clipEncoder;
        _t5Encoder = t5Encoder ?? throw new ArgumentNullException(nameof(t5Encoder));
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image via the primary Qwen2.5-VL path. Token ids come from <c>Qwen2Tokenizer.EncodeChat(prompt, systemPrompt: HunyuanImageQwenTextEncoder.SystemPrompt, addGenerationPrompt: false)</c> padded to <see cref="HunyuanImageQwenTextEncoder.PaddedLength"/> with matching attention masks. Negative inputs may be null when CFG is off (cfg ≤ 1).</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsQwen,
        int[] promptAttentionMaskQwen,
        int[]? negativePromptTokenIdsQwen,
        int[]? negativeAttentionMaskQwen,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        if (_qwenEncoder is null)
            throw new InvalidOperationException(
                "This pipeline was constructed with the legacy CLIP+T5 encoders — use the CLIP/T5 GenerateFromTokens overload.");

        bool useCfg = (request.CfgScale ?? GenerationDefaults.HunyuanImage.CfgScale) > 1.0f;
        if (useCfg && (negativePromptTokenIdsQwen is null || negativeAttentionMaskQwen is null))
            throw new ArgumentException("Negative prompt tokens + mask are required when CfgScale > 1.", nameof(negativePromptTokenIdsQwen));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();

        // Prompt-embedding cache: a hit skips the whole 7B TE phase, which also avoids the TE⇄DiT VRAM swap.
        // The conditioning is per-token only (no resolution-dependent component), so token ids fully key it.
        bool condHit = _cachedCond is not null && _cachedCondKey is not null
            && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIdsQwen);
        bool uncondHit = !useCfg || (_cachedUncond is not null && _cachedUncondKey is not null
            && _cachedUncondKey.AsSpan().SequenceEqual(negativePromptTokenIdsQwen!));
        Tensor cond;
        Tensor? uncond = null;
        if (condHit && uncondHit)
        {
            cond = _cachedCond!;
            if (useCfg) uncond = _cachedUncond;
            Logs.Info("[HunyuanImage] prompt-embedding cache HIT — TE phase skipped.");
        }
        else
        {
            // Cache miss: the DiT cannot share the card with the 7B TE's preload — evict it first.
            // A TE on its OWN device (TextEncoderBackend) never contends with the DiT, so skip the evict
            // (and the re-upload it forces) entirely when split.
            if (_ditResident && ReferenceEquals(TextEncoderBackend, Backend))
            {
                Backend.Sync();
                DitShardBackend?.Sync();
                FreeTransformerWeights();
            }
            TextEncoderBackend.TrimMemoryPool();

            Logs.Info("Encoding text with Qwen2.5-VL-7B (primary 3584-dim per-token)...");
            TextEncoderBackend.PreloadWeights(_qwenEncoder.EnumerateWeights());
            cond = _qwenEncoder.Encode(TextEncoderBackend, promptTokenIdsQwen, promptAttentionMaskQwen);
            uncond = useCfg
                ? _qwenEncoder.Encode(TextEncoderBackend, negativePromptTokenIdsQwen!, negativeAttentionMaskQwen!)
                : null;
            TextEncoderBackend.Sync();
            TextEncoderBackend.FreeWeights(_qwenEncoder.EnumerateWeights());

            // Host-materialize so the cached embeddings survive the activation sweep, then reclaim the
            // encoder's device intermediates before the DiT preload.
            // LOAD-BEARING for TextEncoderDevice placement: these host reads ARE the cross-device boundary —
            // the denoiser's backend re-uploads the conditioning from the host copies.
            _ = cond.DataPointer;
            if (uncond is not null) _ = uncond.DataPointer;
            TextEncoderBackend.FreeActivations();
            TextEncoderBackend.TrimMemoryPool();

            _cachedCond?.Dispose();
            _cachedCond = cond;
            _cachedCondKey = (int[])promptTokenIdsQwen.Clone();
            if (useCfg)
            {
                _cachedUncond?.Dispose();
                _cachedUncond = uncond;
                _cachedUncondKey = (int[])negativePromptTokenIdsQwen!.Clone();
            }
        }

        return Denoise(cond, uncond, request, seed, textOwned: false, onProgress);
    }

    /// <summary>Generates an image from pre-tokenized input via the legacy CLIP+T5 path. CLIP token args are unused (kept for signature stability); the T5 encoder produces per-token context fed to the transformer.</summary>
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
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        if (_t5Encoder is null)
            throw new InvalidOperationException(
                "This pipeline was constructed with the Qwen2.5-VL encoder — use the Qwen GenerateFromTokens overload.");

        bool useCfg = (request.CfgScale ?? GenerationDefaults.HunyuanImage.CfgScale) > 1.0f;
        int seed = request.Seed ?? SeedGenerator.RandomSeed();

        Logs.Info("Encoding text with T5-XXL (legacy per-token context path)...");
        int[][] condTokenBatch = [promptTokenIdsT5];
        int[][]? condMaskBatch = promptAttentionMaskT5 is null ? null : [promptAttentionMaskT5];
        Tensor cond = _t5Encoder.Encode(TextEncoderBackend, condTokenBatch, condMaskBatch);

        Tensor? uncond = null;
        if (useCfg)
        {
            int[][] uncondTokenBatch = [negativePromptTokenIdsT5];
            int[][]? uncondMaskBatch = negativeAttentionMaskT5 is null ? null : [negativeAttentionMaskT5];
            uncond = _t5Encoder.Encode(TextEncoderBackend, uncondTokenBatch, uncondMaskBatch);
        }

        // Split-only hygiene: the single-GPU path never freed here, so keep it byte-identical.
        if (!ReferenceEquals(TextEncoderBackend, Backend))
        {
            TextEncoderBackend.Sync();
            TextEncoderBackend.FreeWeights(_t5Encoder.EnumerateWeights());
            _ = cond.DataPointer;                       // LOAD-BEARING for TextEncoderDevice placement
            if (uncond is not null) _ = uncond.DataPointer;
            TextEncoderBackend.FreeActivations();
        }

        return Denoise(cond, uncond, request, seed, textOwned: true, onProgress);
    }

    /// <summary>Shared denoise + VAE-decode body. The latent stays in patch-token space for the entire loop (patchify once, in-place device <c>CfgEulerStep</c> per step, one unpatchify at the end via the device <see cref="IBackend.UnpatchifyTokens"/> op) so the pipeline glue never forces a per-step D2H round trip — token layout is identical on both sides of the transformer ((py, px, c) inner order), making token-space Euler a pure permutation of the old image-space step. When <paramref name="textOwned"/> is true the conditioning tensors are disposed at the end; the Qwen path passes false (they live in the prompt cache).</summary>
    private (byte[] rgbData, int width, int height, int seed) Denoise(Tensor condText, Tensor? uncondText,
        TextToImageRequest request, int seed, bool textOwned, Action<GenerationProgress>? onProgress)
    {
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.HunyuanImage.Resolve(request);
        const int VaeDownscale = 32;
        if (width % VaeDownscale != 0 || height % VaeDownscale != 0)
            throw new ArgumentException($"Hunyuan Image: width and height must be divisible by {VaeDownscale}.");
        int latentH = height / VaeDownscale;
        int latentW = width / VaeDownscale;
        bool useCfg = cfgScale > 1.0f;
        int patch = _config.PatchSize;
        int inChannels = _config.InChannels;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;

        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && VaeEncoder is null && HyVaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VAE encoder. Set the HyVaeEncoder initializer property (VaeConfig.HunyuanImage).");

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;

        string opMode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "t2i";
        Logs.Info($"Hunyuan Image 2.1 {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        float distilledGuidance = _config.GuidanceEmbed ? cfgScale : 1.0f;

        TensorShape latentShape = new(1, inChannels, latentH, latentW);

        // Initial noise is created in image space (seed parity with the verified e2e recipe) and patchified
        // ONCE — the loop then integrates in token space, so the old per-step patchify/unpatchify host loops
        // (a full D2H drain of the velocity every step) are gone.
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SamplingShift);
        scheduler.SetTimesteps(steps);

        // Masked inpaint (Chroma's blend-on-vanilla pattern): keep the source in TOKEN space plus a token-layout
        // mask, blend after every step so the unmasked region stays on the source's flow trajectory, and
        // recomposite in pixel space at the end. The blend utilities are host ops — touching the latent's
        // DataPointer syncs the device copy down and demotes it, and the next forward re-uploads, so this works
        // against the in-place device CfgEulerStep at a per-step D2H/H2D cost only masked requests pay.
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;
        Tensor? sourceTokens = null;
        Tensor? tokenMask = null;

        Tensor noise = TakeOrCreateNoise(request, latentShape, seed);
        Tensor initialLatent;
        if (request is ImageToImageRequest img2img)
        {
            // Combine in latent space, then patchify once — the same order the text-to-image path uses, so the
            // token layout the loop integrates in is identical either way.
            Tensor sourceLatent = HyVaeEncoder is not null ? HyVaeEncoder.Encode(VaeBackend, img2img.SourceImage)
                : VaeEncoder!.Encode(Backend, img2img.SourceImage);
            initialLatent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(initialLatent, sourceLatent, noise, startStep);
            if (isMaskedInpaint)
            {
                sourceTokens = PatchifyLatent(sourceLatent, inChannels, latentH, latentW, patch);
                // One mask value per TOKEN (the packed grid, latent dims / patch): full precision at patch 1,
                // and at patch 2 the same granularity the token itself has.
                tokenMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH / patch, latentW / patch);
            }
            sourceLatent.Dispose();
        }
        else
        {
            initialLatent = noise;
        }
        Tensor latentTokens = PatchifyLatent(initialLatent, inChannels, latentH, latentW, patch);
        if (!ReferenceEquals(initialLatent, noise)) initialLatent.Dispose();
        noise.Dispose();
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Weight residency for the denoise loop. Two paths, mirroring FluxPipeline:
        //   - StreamingCache != null (CUDA): pick eager-vs-streaming by measured free VRAM. When the whole
        //     60-block set fits beside the activation reserve we preload everything (unchanged behavior —
        //     the sliding window re-uploads every block on every forward, which is pure loss on a card that
        //     never needed it). Otherwise a BlockStreamingController caps resident weights at ~(prefetch+2)
        //     blocks, which is what lets the 17B DiT run on 12 GB instead of OOMing.
        //   - StreamingCache == null (CPU/Vulkan): eager preload, exactly as before.
        BlockStreamingController? streamer = null;
        if (DitShardBackend is not null)
        {
            // Sharding beats streaming: the pooled two-card VRAM makes both split ranges resident, which is
            // strictly better than sliding-window re-uploads. Asymmetric preload — shared + [0, split) on the
            // primary, ONLY [split, BlockCount) on the shard backend; EnumerateWeights() on both would
            // replicate instead of pool.
            if (Backend.StreamingCache is not null)
                Logs.Info("HunyuanImage: DiT sharding overrides low-VRAM block streaming for this generation.");
            if (!_ditResident)
            {
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                DitShardBackend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
            }
            _ditResident = true;
        }
        else if (Backend.StreamingCache is not null)
        {
            IStreamingBlock[] blocks = new IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);

            int txtSeqLen = (int)Math.Max(condText.Shape[1], uncondText?.Shape[1] ?? 0);
            int imgSeqLen = hPacked * wPacked;
            long activationReserve = EstimateActivationReserveBytes(
                txtSeqLen, imgSeqLen, _config.HiddenSize, (int)(_config.HiddenSize * _config.MlpRatio))
                + WeightBytes.Sum(_transformer.EnumerateSharedWeights());

            long totalBlockBytes = 0;
            foreach (IStreamingBlock block in blocks) totalBlockBytes += block.EstimatedWeightBytes;
            // The resident-vs-streamed decision — including the already-resident short-circuit that stops warm
            // generations oscillating resident→streaming→resident — belongs to VramPlanner, so every pipeline
            // decides identically and HARTSY_LOWVRAM can override it. The reserve estimate above stays here:
            // it is genuinely per-architecture, unlike the decision.
            VramPlanner planner = new VramPlanner(Backend.StreamingCache, "HunyuanImage", Backend);
            // Forced streaming has to displace a warm DiT before the planner measures, or the already-resident
            // short-circuit answers Resident and the setting does nothing on exactly the generations it was set for.
            if (planner.ShouldDisplaceResident(_ditResident))
            {
                Backend.Sync();
                FreeTransformerWeights();
            }
            PhasePlacement placement = planner.PlanPhase(
                "denoise", totalBlockBytes, activationReserve, alreadyResident: _ditResident, canStream: true);
            if (placement == PhasePlacement.Resident)
            {
                if (!_ditResident) Backend.PreloadWeights(_transformer.EnumerateWeights());
                _ditResident = true;
            }
            else
            {
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                // Re-queries free VRAM after the shared preload while activationReserve still includes the
                // shared bytes, so those are counted twice. Strictly conservative, and inert against the
                // clamp-at-2 cap — not worth threading a second reserve value through.
                int prefetchAhead = ChoosePrefetchAhead(Backend.StreamingCache, blocks, activationReserve);
                streamer = new BlockStreamingController(Backend.StreamingCache, blocks,
                    prefetchAhead: prefetchAhead, retainBehind: 0, backend: Backend);
                _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
                streamer.Prime();
                long totalMb = streamer.EstimatedTotalWeightBytes / (1024 * 1024);
                Logs.Info($"Hunyuan Image streaming: {blocks.Length} blocks, prefetchAhead={prefetchAhead}, " +
                    $"total ~{totalMb} MB, activation reserve ~{activationReserve / (1024 * 1024)} MB");
            }
        }
        else if (!_ditResident)
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
        }

        // Sampler selection (2026-08-20). Hunyuan Image is flow-matching, so it had no user-selectable sampler at all
        // before this. The sampler owns the integrator and the sigma spacing; everything family-specific — the
        // distilled-guidance embedding, DiT sharding and block streaming, all of which live inside RunForward — stays
        // in the predictor closure below. There is no step graph, no step cache and no host/reference loop on this
        // path (masked inpaint blends against the same in-place device step), so nothing else needs narrowing for a
        // non-default sampler.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Hunyuan Image",
            startsFromNoisedInit: startStep > 0);

        // The predictor closure needs the timestep table, and `timesteps` is a ReadOnlySpan (a ref local) that a
        // lambda cannot capture. One small array copy per generation.
        float[] timestepTable = timesteps.ToArray();

        Logs.Info($"Starting denoise loop ({steps} steps, distilled guidance={distilledGuidance})...");
        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();

            // CFG combine + Euler stay ONE in-place device op (CfgEulerStep: z += (neg + gw·(pos−neg))·dt — for
            // gw=cfgScale this IS uncond + cfg·(cond−uncond); gw=1 degenerates to the plain Euler step), keeping the
            // latent tokens device-resident across the whole loop: EulerSampler.Step IS that same fused call.
            DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                PredictionType.FlowVelocity,
                (x, s, stepIndex) =>
                {
                    // Unlike the rest of the DiT fleet, Hunyuan's transformer takes the timestep on the PUBLIC 0-1000
                    // conditioning scale, not the [0,1] sigma. On-schedule steps therefore reuse the table entry
                    // verbatim — `sigma·1000` is not an exact F32 round trip back to the value SetTimesteps stored, so
                    // recomputing it would shift every existing Hunyuan generation by an ulp of conditioning. Only a
                    // genuine off-schedule sub-step (what a second-order sampler evaluates at) scales the raw sigma,
                    // which is right: there is no precomputed timestep for it.
                    float stepT = stepIndex < steps && s == scheduler.SigmaAt(stepIndex) ? timestepTable[stepIndex]
                        : s * 1000.0f;
                    Tensor cond = RunForward(x, condText, stepT, distilledGuidance, hPacked, wPacked);
                    if (!useCfg || _config.GuidanceEmbed)
                    {
                        // Distilled guidance rides in the embedding, so there is no unconditional branch and the
                        // combine must be exactly 1.0 — substituting cfgScale would not cancel bit-exactly in F32.
                        return new DenoisePrediction(cond, cond);
                    }
                    // Standard CFG path (un-distilled variant): hand back the raw pair and let the fused device op
                    // do the uncond-anchored combine, exactly as the direct call did.
                    Tensor uncond = RunForward(x, uncondText!, stepT, 1.0f, hPacked, wPacked);
                    return new DenoisePrediction(cond, uncond, cfgScale);
                });
            if (i == startStep)
            {
                sampler.Reset(latentTokens.Shape);
            }
            sampler.Step(Backend, latentTokens, predictor, i);

            // Masked-inpaint blend: re-noise the source tokens at the next step's sigma and paste them over the
            // unmasked region. Final step blends the clean source (no denoising follows it).
            if (isMaskedInpaint && sourceTokens is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNchw = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    Tensor freshTokens = PatchifyLatent(freshNchw, inChannels, latentH, latentW, patch);
                    freshNchw.Dispose();
                    noisedSource = new Tensor(sourceTokens.Shape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourceTokens, freshTokens, nextStep);
                    freshTokens.Dispose();
                }
                else
                {
                    noisedSource = sourceTokens;
                }
                MaskBlendUtilities.BlendTokensInPlace(latentTokens, noisedSource, tokenMask!);
                if (!ReferenceEquals(noisedSource, sourceTokens)) noisedSource.Dispose();
            }

            stepSw.Stop();
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }
        sourceTokens?.Dispose();
        tokenMask?.Dispose();

        if (textOwned)
        {
            condText.Dispose();
            uncondText?.Dispose();
        }

        Backend.Sync();
        DitShardBackend?.Sync();
        // Tear down before the unpatchify + 32-channel VAE decode — that decode is what needs the room back.
        _transformer.BeforeBlockForward = null;
        if (streamer is not null)
        {
            streamer.EvictAll();
            streamer.Dispose();
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            // Also purge the block weights: the streaming cache registers each uploaded block in the backend
            // weight cache, so any block still cached at the end would stay resident and starve the decode.
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        else if (!KeepModelsResident)
        {
            FreeTransformerWeights();
        }

        // Device unpatchify (innerChannelFastest: (py, px, c) inner order — matches PatchifyLatent) keeps the
        // latent → VAE chain device-resident; the CPU backend falls back to the IBackend host loop.
        Tensor latent = new(latentShape, DType.F32);
        Backend.UnpatchifyTokens(latent, latentTokens, inChannels, hPacked, wPacked, patch, innerChannelFastest: true);
        latentTokens.Dispose();

        if (!ReferenceEquals(VaeBackend, Backend))
        {
            // LOAD-BEARING for VaeDevice: force the device-resident latent to host so the VAE backend
            // uploads a valid copy — one small D2H vs the multi-second decode.
            Backend.Sync();
            _ = latent.DataPointer;
        }

        Logs.Info("Decoding latents to image (32-channel VAE)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor decoded = _hyVaeDecoder is not null ? _hyVaeDecoder.Decode(VaeBackend, latent)
            : _vaeDecoder.Decode(VaeBackend, latent);
        latent.Dispose();
        vaeSw.Stop();

        // Pixel-space recomposite for masked inpaint: paste decoded over the source where mask=1, so the
        // unmasked region is byte-exact source rather than a VAE roundtrip of it.
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(decoded, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(decoded);
        decoded.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI) the decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the next generation's budget — and under KEEP_MODELS the
        // trimmed pool is what lets a cross-model switch reclaim VRAM beside the resident DiT.
        Backend.FreeActivations();
        if (!ReferenceEquals(VaeBackend, Backend)) VaeBackend.FreeActivations();

        sw.Stop();
        Logs.Info($"Hunyuan Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgb, width, height, seed);
    }

    /// <summary>Routes the per-step forward to <see cref="HunyuanImageTransformer.ForwardSharded"/> when DiT sharding is configured and the plain <see cref="HunyuanImageTransformer.Forward"/> otherwise, keeping the denoise-loop body identical either way. The ByT5 glyph stream is not wired yet, so both paths pass a null <c>encoderHidden2</c>.</summary>
    private Tensor RunForward(Tensor latentTokens, Tensor text, float t, float guidance, int hPacked, int wPacked) =>
        DitShardBackend is not null
            ? _transformer.ForwardSharded(Backend, DitShardBackend, latentTokens, text, encoderHidden2: null,
                t, guidance, hPacked, wPacked, DitShardSplitBlock)
            : _transformer.Forward(Backend, latentTokens, text, encoderHidden2: null, t, guidance, hPacked, wPacked);

    /// <summary>Frees the DiT weights on whichever backend(s) hold them — the mirror of the sharded asymmetric preload. The unsharded <c>FreeWeights(EnumerateWeights())</c> would silently no-op on the shard backend's block range (frees are per-backend) and leak it every non-resident generation.</summary>
    private void FreeTransformerWeights()
    {
        if (DitShardBackend is not null)
        {
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
        }
        else
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
        }
        _ditResident = false;
    }

    /// <summary>Estimates the peak per-forward activation footprint (bytes) that must stay free beside the DiT weights, so the resident-vs-stream decision and the prefetch depth can never disagree. The deepest valley is a single-stream block holding the joint <c>[img,txt]</c> stream, the modulated copy, Q/K/V both pre- and post-permute, and the parallel MLP branch (proj + activated + the feature-dim concat) at once. Everything is F32: this transformer deliberately does not opt into the shared 16-bit hot path (its Qwen2.5-VL text stream overflows F16 — see the comment in <see cref="HunyuanImageTransformer.Forward"/>). Sized for B=1.
    /// <para>Kept private rather than shared with <c>FluxPipeline</c>'s equivalent: the tensor inventory and the
    /// activation dtype are both per-architecture, and the two would only diverge if merged.</para></summary>
    private static long EstimateActivationReserveBytes(int txtSeqLen, int imgSeqLen, int hiddenSize, int mlpDim)
    {
        long totalSeqLen = txtSeqLen + imgSeqLen;
        long mlpProjBytes = totalSeqLen * mlpDim * 4;
        long mlpActivatedBytes = totalSeqLen * mlpDim * 4;
        long concattedBytes = totalSeqLen * (hiddenSize + mlpDim) * 4;
        long jointBytes = totalSeqLen * hiddenSize * 4;
        long modulatedBytes = totalSeqLen * hiddenSize * 4;
        long attnFlatBytes = totalSeqLen * hiddenSize * 4;
        long qkvBytes = 3L * totalSeqLen * hiddenSize * 4;
        long mhBytes = 3L * totalSeqLen * hiddenSize * 4;
        long scratchBytes = mlpProjBytes + mlpActivatedBytes + concattedBytes + jointBytes
            + modulatedBytes + attnFlatBytes + qkvBytes + mhBytes;
        // Fudge for cuBLAS workspace, GGUF dequant buffers, and the cached RoPE tables. The shared (non-block)
        // weights are NOT folded in here — the caller adds their measured size, because the Hunyuan token
        // refiner makes that set far too large to hide inside a constant.
        return scratchBytes + 1024L * 1024 * 1024;
    }

    /// <summary>Picks the in-flight prefetch depth that keeps the streaming peak inside the free-VRAM budget.</summary>
    private static int ChoosePrefetchAhead(IStreamingWeightCache cache, IStreamingBlock[] blocks, long activationReserve)
    {
        long avail = cache.QueryAvailableWeightCacheBytes(activationReserve);
        if (avail <= 0) return 0;

        // Size against the LARGEST block, not block 0: the 20 double-stream blocks carry two full
        // attention+FFN streams and are roughly twice a single-stream block.
        long perBlockBytes = 0;
        foreach (IStreamingBlock block in blocks) perBlockBytes = Math.Max(perBlockBytes, block.EstimatedWeightBytes);
        if (perBlockBytes <= 0) return 1;

        // The working set briefly hits (prefetchAhead + 2) blocks when block N+1 is made resident before
        // block N-1 is evicted. Cap at 2 — deeper burns VRAM without hiding more latency.
        int maxByBudget = (int)(avail / perBlockBytes) - 2;
        return Math.Clamp(maxByBudget, 0, 2);
    }

    /// <summary>Patchifies <c>[B, C, H, W]</c> -&gt; <c>[B, S, p²·C]</c> with channel-outer ordering inside each patch (matches the diffusers <c>einops.rearrange("B C (H p) (W q) -&gt; B (H W) (p q C)")</c>).</summary>
    private static Tensor PatchifyLatent(Tensor latent, int channels, int height, int width, int patch)
    {
        int batch = (int)latent.Shape[0];
        int hPacked = height / patch;
        int wPacked = width / patch;
        int seqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        TensorShape outShape = new(batch, seqLen, patchVolume);
        Tensor result = new(outShape, DType.F32);

        float* src = (float*)latent.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + b * chwStride;
            float* batchDst = dst + (long)b * seqLen * patchVolume;
            for (int hp = 0; hp < hPacked; hp++)
            {
                for (int wp = 0; wp < wPacked; wp++)
                {
                    long tokenIdx = (long)hp * wPacked + wp;
                    float* tokenDst = batchDst + tokenIdx * patchVolume;
                    int outIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int srcRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int srcCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                                tokenDst[outIdx++] = batchSrc[c * hwStride + srcRow * width + srcCol];
                        }
                    }
                }
            }
        }
        return result;
    }

}
