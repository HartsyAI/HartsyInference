using HartsyInference.Diffusion.Sampling;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Krea 2 text-to-image pipeline. Encodes the prompt through Qwen3-VL-4B (12-layer hidden-state tap) via <see cref="LlamaStyleEncoder.EncodeMultiLayer"/>, runs the <see cref="Krea2Transformer"/> (which fuses the tapped layers, patchifies the latent, and predicts the flow-match velocity) under a resolution-aware flow-match Euler schedule, and decodes through the 16-channel Qwen-Image VAE. CFG is a dual pass when <c>request.CfgScale &gt; 1</c> (Base, ~4.5); Turbo/TDM runs a single conditional pass (guidance off). See <c>docs/Research/KREA2.md</c>.</summary>
public sealed class Krea2Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Krea2Transformer _transformer;
    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop FreeWeights + next-gen ~1.8 s re-upload). The TE is still freed each gen — its VRAM is needed by the VAE decode (see the call site). Requires DiT + VAE-decode peak to fit VRAM (fp8 Krea2 on 24 GB: yes). Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables) — the miss-path eviction above is what keeps smaller cards viable even with residency on.</summary>
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);

    /// <summary>Whether the DiT's weights are currently ALL on the device from a previous generation (<see cref="KeepModelsResident"/>). Fed to <see cref="VramPlanner.PlanPhase"/> as <c>alreadyResident</c> — the availability query cannot see past weights that are themselves occupying the space it measures, so without this a warm generation reports "does not fit" and flips between resident and streamed on alternate runs.</summary>
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used): tapped TE hidden states keyed on
    // (token ids, drop index). See the encode block in GenerateFromTokens.
    private int[]? _cachedCondKey;
    private int _cachedCondDrop = -1;
    private Tensor? _cachedCond;
    private int[]? _cachedUncondKey;
    private int _cachedUncondDrop = -1;
    private Tensor? _cachedUncond;

    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly Krea2Config _config;

    /// <summary>Creates a Krea 2 pipeline. The caller owns each component's lifetime (they may be shared/reused). Img2img is unavailable; use the overload accepting a <see cref="QwenImageVaeEncoder"/> to enable it.</summary>
    public Krea2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Krea2Transformer transformer,
        QwenImageVaeDecoder vaeDecoder, Krea2Config config)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Krea 2 pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>).</summary>
    public Krea2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Krea2Transformer transformer,
        QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder, Krea2Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Generates an image. <paramref name="promptTokenIds"/> / <paramref name="negativeTokenIds"/> are the chat-templated Qwen token sequences; the leading <paramref name="promptDropIndex"/> (system-prefix) hidden states are dropped (Krea 2's <c>prompt_template_encode_start_idx = 34</c>). The negative stream is used only when <c>request.CfgScale &gt; 1</c> (Base); for Turbo pass <c>CfgScale ≤ 1</c>.
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded through the Qwen-Image
    /// 3D-causal VAE encoder and noised via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a
    /// <see cref="QwenImageVaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint
    /// (per-step latent blend + final pixel recomposite, same as Z-Image). Strength=0 short-circuits to byte-identical
    /// pass-through.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[]? negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 34,
        int negativeDropIndex = 34,
        RegionalPlan? regionalPlan = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a QwenImageVaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Generic.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imageSeqLen = hPacked * wPacked;
        bool useCfg = cfgScale > 1.0f && negativeTokenIds is not null;

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
                      : "t2i";
        Logs.Info($"Krea 2 {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}, distilled={_config.IsDistilled}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── encode prompt: 12-layer tap → [1, S, 12·2560], drop the system prefix ──
        // Prompt-embedding cache: identical (tokens, dropIndex) reuse the previous gen's hidden states — the
        // whole TE phase (preload + encode + free ≈ 1.1 s) vanishes for repeat prompts (seed-only changes),
        // matching ComfyUI's conditioning cache. Reusing the SAME tensor reference also keeps the
        // transformer's txt-fusion cache warm (it's keyed on the encoderHidden reference). The cached tensor
        // is host-materialized (DataPointer read frees its GPU buffer) so it survives activation frees; each
        // step's use re-uploads ~5 MB, which the weight auto-promotion then pins. Cache holds one cond + one
        // uncond (the last used); a new prompt evicts + disposes.
        long phT0 = sw.ElapsedMilliseconds;
        bool condHit = _cachedCond is not null && _cachedCondDrop == promptDropIndex
            && _cachedCondKey is not null && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIds);
        bool uncondHit = !useCfg || (_cachedUncond is not null && _cachedUncondDrop == negativeDropIndex
            && _cachedUncondKey is not null && _cachedUncondKey.AsSpan().SequenceEqual(negativeTokenIds!));
        Tensor condHidden;
        Tensor? uncondHidden;
        long phT1, phT2;
        if (condHit && uncondHit)
        {
            condHidden = _cachedCond!;
            uncondHidden = useCfg ? _cachedUncond : null;
            phT1 = phT2 = sw.ElapsedMilliseconds;
        }
        else
        {
            Backend.PreloadWeights(_textEncoder.EnumerateWeights());
            phT1 = sw.ElapsedMilliseconds;
            condHidden = condHit ? _cachedCond! : EncodeTapped(promptTokenIds, promptDropIndex);
            uncondHidden = useCfg ? (uncondHit ? _cachedUncond : EncodeTapped(negativeTokenIds!, negativeDropIndex)) : null;
            phT2 = sw.ElapsedMilliseconds;
            if (!condHit)
            {
                unsafe { _ = condHidden.DataPointer; }   // materialize to host so it survives across gens
                if (!ReferenceEquals(_cachedCond, condHidden)) _cachedCond?.Dispose();
                _cachedCond = condHidden;
                _cachedCondKey = (int[])promptTokenIds.Clone();
                _cachedCondDrop = promptDropIndex;
            }
            if (useCfg && !uncondHit)
            {
                unsafe { _ = uncondHidden!.DataPointer; }
                if (!ReferenceEquals(_cachedUncond, uncondHidden)) _cachedUncond?.Dispose();
                _cachedUncond = uncondHidden;
                _cachedUncondKey = (int[])negativeTokenIds!.Clone();
                _cachedUncondDrop = negativeDropIndex;
            }
        }
        // The TE is ALWAYS freed after encode, even under HARTSY_KEEP_MODELS: its ~4-8 GB is exactly the
        // headroom the VAE decode's im2col needs at 1024² (keeping TE+DiT+VAE resident OOM'd: the decode
        // requested 6.9 GB with 69 MB free). Re-encoding costs ~1 s/gen; keeping the 13 GB DiT saves ~2 s.
        if (!(condHit && uncondHit))
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
        long phT3 = sw.ElapsedMilliseconds;
        Logs.Verbose($"[krea2-phase] TE preload={phT1 - phT0}ms encode={phT2 - phT1}ms free={phT3 - phT2}ms"
            + (condHit && uncondHit ? " (cache hit)" : ""));

        // ── Regional conditioning (Tier 3.7): append region text streams to the COND stream only — matches
        // FluxPipeline's precedent (true-CFG's negative pass runs against the unextended stream; regions are a
        // positive-conditioning-only concept). Image tokens follow text in Krea 2's joint [txt|img] concat
        // (ForwardEmbedIn: `backend.Concat(joint, new[] { txt, img }, dim: 1)`), same layout Flux/Flux.2 use. ──
        bool hasRegions = regionalPlan is not null && regionalPlan.Regions.Count > 0;
        Tensor? extendedCond = null;
        List<(int Start, int End)>? regionRanges = null;
        List<float[]>? regionGridMasks = null;
        float[]? regionWeights = null;
        int condTxtSeqLen = (int)condHidden.Shape[1];
        if (hasRegions)
        {
            (extendedCond, condTxtSeqLen, regionRanges, regionGridMasks) =
                RegionalConditioningLayout.BuildTextStream(regionalPlan!, condHidden, hPacked, wPacked);
            regionWeights = new float[regionalPlan!.Regions.Count];
            if (DitShardBackend is not null)
            {
                Logs.Warning("[Krea2Pipeline] Regional prompting is active — DiT sharding v1 has no attnBias "
                    + "surface (ForwardPatchedSharded), running this generation unsharded on the primary backend.");
            }
        }

        // ── scheduler: resolution-aware exp shift (Turbo pins mu=1.15) ──
        FlowMatchEulerDiscreteScheduler scheduler = _config.IsDistilled
            ? new FlowMatchEulerDiscreteScheduler(MathF.Exp(1.15f))
            : FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen, baseSeqLen: 256, maxSeqLen: 6400,
                baseShift: 0.5f, maxShift: 1.15f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        TensorShape latentShape = new(1, _config.VaeChannels, latentH, latentW);
        (Tensor latent, Tensor? sourceLatent) =
            BuildInitialLatent(request, scheduler, latentShape, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? latentMask = null;
        if (isMaskedInpaint)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
        }

        // Krea 2 is a 12.9B single-stream MMDiT whose shipped fp8_scaled checkpoint stays packed fp8 on device
        // (~13 GB of weights, ~11 GB of it in the 28 blocks). That does not fit a 12 GB card at all — measured
        // 2026-07-27, a fresh process OOM'd at an 11,709 MiB peak. Block streaming gives the denoise loop the same
        // graceful degradation Ideogram 4 / Qwen-Image got: the shared weights stay resident and the blocks ride a
        // sliding window. Cards with real headroom take the planner's Resident branch and behave exactly as before.
        long phT4 = sw.ElapsedMilliseconds;
        BlockStreamingController? streamer = null;
        if (DitShardBackend is not null)
        {
            // Phase 8 DiT sharding: asymmetric preload is the whole point — Backend gets the shared weights PLUS
            // its block range, DitShardBackend gets ONLY its block range. Preloading EnumerateWeights() on both
            // would replicate the DiT (the CFG-parallel mistake this feature exists to avoid), defeating VRAM
            // pooling. Not composable with block streaming (BeforeBlockForward IS the split mechanism) — sharding
            // takes priority even if the backend's LowVram mode would otherwise stream, since sharding already
            // solves the "doesn't fit on one card" problem this generation is facing.
            if (Backend.StreamingCache is not null)
            {
                Logs.Info("Krea2 DiT sharding active — overriding low-VRAM block streaming " +
                    "(sharding pools VRAM across both backends instead).");
            }
            Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
            Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
            _ditResident = true;
        }
        else if (Backend.StreamingCache is not null)
        {
            IStreamingBlock[] blocks = new IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);
            long totalBlockBytes = 0;
            foreach (IStreamingBlock block in blocks) totalBlockBytes += block.EstimatedWeightBytes;
            long sharedBytes = WeightBytes.Sum(_transformer.EnumerateSharedWeights());
            int txtSeqLen = (int)condHidden.Shape[1];
            long reserve = EstimateActivationReserveBytes(txtSeqLen, imageSeqLen, _config) + sharedBytes;

            VramPlanner planner = new VramPlanner(Backend.StreamingCache, "Krea2", Backend);
            // Forced streaming has to displace a warm DiT before the planner measures, or the already-resident
            // short-circuit answers Resident and the setting does nothing on exactly the generations it was set for.
            // Frees per-backend like the post-denoise path: an unsharded free would no-op on the shard's range.
            if (planner.ShouldDisplaceResident(_ditResident))
            {
                Backend.Sync();
                _transformer.InvalidateStepGraph(Backend);
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
            PhasePlacement placement = planner.PlanPhase(
                "denoise", totalBlockBytes, reserve, alreadyResident: _ditResident, canStream: true);
            if (placement == PhasePlacement.Resident)
            {
                // Unconditional even when already resident — byte-for-byte the pre-streaming behavior. Per-weight
                // preload is a cache lookup for anything already on the device, so this costs nothing on a warm
                // generation and self-heals if the cache dropped a weight behind our back (which _ditResident,
                // being only a planner hint, would not notice).
                Backend.PreloadWeights(_transformer.EnumerateWeights());
                _ditResident = true;
            }
            else
            {
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                long avail = Backend.StreamingCache.QueryAvailableWeightCacheBytes(reserve);
                long perBlock = blocks.Length > 0 ? blocks[0].EstimatedWeightBytes : 0;
                int prefetchAhead = perBlock > 0 ? Math.Clamp((int)(avail / perBlock) - 2, 0, 2) : 0;
                streamer = new BlockStreamingController(Backend.StreamingCache, blocks, prefetchAhead: prefetchAhead, retainBehind: 0);
                _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
                streamer.Prime();
                _ditResident = false;
                Logs.Info($"Krea2 streaming: {blocks.Length} blocks, prefetchAhead={prefetchAhead}, " +
                    $"total ~{totalBlockBytes / (1024 * 1024)} MB, shared ~{sharedBytes / (1024 * 1024)} MB resident, " +
                    $"reserve ~{reserve / (1024 * 1024)} MB");
            }
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
        }
        Logs.Verbose($"[krea2-phase] DiT preload={sw.ElapsedMilliseconds - phT4}ms");

        // Fast path (plain t2i, no img2img / masked-inpaint): keep the latent in the transformer's patchified token
        // space across the WHOLE sampling loop — patchify once here, run each step's flow-match Euler update
        // (x += v·dt) on-device (Scale+Add), unpatchify once after. A denoise step then never reads a tensor's
        // DataPointer, so the host queues all steps without the per-step D2H pipeline drain that serialized host
        // dispatch against GPU execution (the dominant host-bound cost). Img2img / inpaint keep the pixel-space path.
        bool fastPath = !isImg2Img && !isMaskedInpaint;
        // Default-off across-step First-Block cache (HARTSY_STEP_CACHE + optional HARTSY_STEP_CACHE_LATE —
        // reference wiring QwenImagePipeline / Ideogram4Pipeline; INFERENCE_ACCEL_GRIND §H1.5). An armed cache
        // is per-step-variable topology, so it forces the eager path (no step-graph capture).
        // Calibrated ship point (Turbo only — Base is unmeasured, so it gets no profile and =1 stays the
        // generic raw 0.10): raw 0.15 in the LATE half — 1.13× at SSIM 0.9740 (U-shaped drift, ONE safe
        // late reuse at 8 distilled steps; results doc 2026-07-22_accel_stepcache_krea2turbo_4090.md).
        StepCacheProfile? profile = _config.IsDistilled
            ? new StepCacheProfile(Threshold: 0.15f, Cap: 3, Poly: null, LateWindow: 0.5f) : null;
        (float stepCacheThreshold, int stepCacheCap, float[]? stepCachePoly, float stepCacheLate) =
            StepCacheEnv.Resolve(profile);
        DeviceFeatureCache? condCache = null;
        DeviceFeatureCache? uncondCache = null;
        if (stepCacheThreshold > 0f)
        {
            if (DitShardBackend is not null)
            {
                Logs.Warning("HARTSY_STEP_CACHE set but DiT sharding is active — step-cache's block-0-indicator " +
                    "shape doesn't compose with a fixed block-range boundary (see ForwardPatchedSharded); running uncached.");
            }
            else if (Backend.SupportsDeviceStepCacheGate)
            {
                condCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                if (useCfg) uncondCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}"
                    + (stepCacheLate > 0f ? $", lateWindow={stepCacheLate}" : ""));
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }
        // Step-graph mode (HARTSY_DIT_GRAPH, fast path only, no CFG): route the latent through the
        // transformer's FIXED buffer so the captured graph's baked address stays valid across steps and gens.
        // The fixed tensor is transformer-owned: never disposed here, never DataPointer-read (snapshot instead).
        // NEVER on the streamed path: the transformer refuses to capture while BeforeBlockForward is hooked (a graph
        // bakes weight pointers that streaming re-points every forward), so it returns a fresh eager velocity — and
        // the graph-mode branch below skips disposing it, which would leak one velocity buffer per step.
        // Sampler selection (2026-08-20). Krea 2 is flow-matching, so it had no user-selectable sampler at all before
        // this — the whole DiT fleet was Euler-only. The sampler owns the integrator and the sigma spacing; everything
        // family-specific (cond-anchored CFG, region bias, the step-cache gates) stays in the predictor closure below.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Krea 2");
        bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);

        // The host/reference branch below does not consult the sampler at all, so a non-default selection there would
        // be silently dropped — the exact failure this whole change removes. Refuse by name instead.
        if (!fastPath && nonDefaultSampler)
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' runs only on Krea 2's drain-free path, and this generation "
                + "fell back to the reference loop (img2img or masked inpaint). Drop the sampler selection, or drop the feature that "
                + "forced the fallback.");
        }

        // The predictor closure needs the timestep table, and `timesteps` is a ReadOnlySpan (a ref local) that a
        // lambda cannot capture. One small array copy per generation.
        float[] timestepTable = timesteps.ToArray();

        // A non-default sampler is excluded from step-graph capture, on the same grounds the gate already excludes
        // CFG, step-cache, streaming, sharding and regions: a captured graph bakes one fixed sequence of ops at fixed
        // addresses, and a second-order sampler runs an EXTRA forward at an intermediate sigma plus scratch-tensor
        // arithmetic between them. Narrowing for that generation is the established precedent here; silently capturing
        // the wrong sequence is not.
        bool graphMode = fastPath && !useCfg && condCache is null && streamer is null && DitShardBackend is null && !hasRegions
            && !nonDefaultSampler && Models.Denoisers.DiTBlocks.DitStepGraph.Enabled && Backend.StepGraphSupported;
        Tensor? patchLatent = null;
        if (fastPath)
        {
            Tensor fresh = _transformer.PatchifyLatent(latent);
            if (graphMode)
            {
                patchLatent = _transformer.PrepareGraphLatent(Backend, fresh);
                fresh.Dispose();
            }
            else
            {
                patchLatent = fresh;
            }
        }

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i] / 1000.0f; // scheduler stores sigma·1000; transformer takes t∈[0,1]

            // Late-window cache gate (HARTSY_STEP_CACHE_LATE): reuse eligible only in the schedule tail —
            // early-schedule reuse is where quality damage concentrates (Ideogram 4 results doc).
            bool cacheEligible = stepCacheLate <= 0f || (i + 1) > steps * (1f - stepCacheLate);
            DeviceFeatureCache? stepCondCache = cacheEligible ? condCache : null;
            DeviceFeatureCache? stepUncondCache = cacheEligible ? uncondCache : null;

            if (fastPath)
            {
                if (graphMode)
                {
                    // Step-graph capture keeps its own direct call. `v` here is the transformer's FIXED velocity
                    // buffer, reused every step and never disposed, and the captured graph bakes that address — so
                    // this path deliberately does NOT route through the sampler, whose scratch tensors and (for
                    // second-order methods) extra forward would invalidate the capture. The gate above already
                    // excludes every sampler but Euler from reaching here.
                    Tensor graphV = RunForwardPatched(patchLatent!, t, condHidden, hPacked, wPacked, stepCondCache);
                    Backend.CfgEulerStep(patchLatent!, graphV, graphV, 1.0f, scheduler.Dt(i));
                }
                else
                {
                    // Everything family-specific lives in the closure: Krea 2 applies its own cond-anchored CFG
                    // (cond + scale·(cond − uncond), NOT the linear uncond-anchored combine), so the combined
                    // velocity is returned as both halves of the pair with guidance 1.0 — making the sampler's own
                    // combine an identity rather than double-applying guidance.
                    DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                        PredictionType.FlowVelocity,
                        (x, sigma, stepIndex) =>
                        {
                            // The transformer takes t ∈ [0,1], which for a flow-match family IS the sigma (the
                            // scheduler's public Timesteps scale is just sigma·1000).
                            //
                            // On-schedule sigmas reuse the precomputed `timesteps[i]/1000` expression the loop used
                            // before the sampler seam, rather than passing sigma straight through. Those are the same
                            // number mathematically, but `(s·1000f)/1000f` is not an exact round trip in F32 — 1000 is
                            // not a power of two — so substituting one for the other would perturb every existing
                            // Krea 2 generation by an ulp of conditioning. Only an off-schedule sub-step (what a
                            // second-order sampler evaluates at) takes the raw sigma, which is exactly right: there is
                            // no precomputed timestep for it.
                            float t = stepIndex < steps && sigma == scheduler.SigmaAt(stepIndex)
                                ? timestepTable[stepIndex] / 1000.0f : sigma;
                            Tensor? bias = null;
                            if (hasRegions)
                            {
                                regionalPlan!.ResolveStep(stepIndex - startStep, regionWeights!);
                                bias = RegionalAttentionBias.Build(
                                    condTxtSeqLen + imageSeqLen, condTxtSeqLen, imageSeqLen, regionRanges!, regionGridMasks!, regionWeights!);
                            }
                            try
                            {
                                Tensor combined;
                                if (useCfg)
                                {
                                    // The negative pass runs against the unextended uncondHidden, no bias — same
                                    // convention FluxPipeline uses (regions are positive-conditioning-only).
                                    Tensor condV = RunForwardPatched(x, t, hasRegions ? extendedCond! : condHidden, hPacked, wPacked, stepCondCache, bias);
                                    Tensor uncondV = RunForwardPatched(x, t, uncondHidden!, hPacked, wPacked, stepUncondCache);
                                    combined = CfgHelper.ApplyCfgCondAnchored(condV, uncondV, cfgScale);
                                    uncondV.Dispose();
                                    condV.Dispose();
                                }
                                else
                                {
                                    combined = RunForwardPatched(x, t, hasRegions ? extendedCond! : condHidden, hPacked, wPacked, stepCondCache, bias);
                                }
                                return new DenoisePrediction(combined, combined);
                            }
                            finally
                            {
                                bias?.Dispose();
                            }
                        });
                    if (i == startStep)
                    {
                        sampler.Reset(patchLatent!.Shape);
                    }
                    sampler.Step(Backend, patchLatent!, predictor, i);
                }

                stepSw.Stop();
                Logs.Verbose($"[krea2-phase] step {i + 1}/{steps}: {stepSw.ElapsedMilliseconds}ms");
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
                TrimPoolOnStreamedPath(streamer);
                continue;
            }

            Tensor noisePred;
            if (useCfg)
            {
                Tensor cond = RunForward(latent, t, condHidden);
                Tensor uncond = RunForward(latent, t, uncondHidden!);
                // VALIDATION-PENDING: Krea 2 uses cond-anchored CFG (cond + scale*(cond - uncond)); guidance_scale=4.5 here ≈ 5.5 under the standard uncond-anchored convention. Verify vs reference.
                noisePred = CfgHelper.ApplyCfgCondAnchored(cond, uncond, cfgScale);
                uncond.Dispose();
                cond.Dispose();
            }
            else
            {
                noisePred = RunForward(latent, t, condHidden);
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
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
            TrimPoolOnStreamedPath(streamer);
        }

        // Fast path: bring the final patchified latent back to pixel space once for the VAE. Graph mode reads a
        // SNAPSHOT — unpatchify's DataPointer read would otherwise D2H-and-FREE the fixed buffer the graph bakes.
        if (fastPath)
        {
            latent.Dispose();
            Tensor tokens = graphMode ? _transformer.SnapshotGraphLatent(Backend) : patchLatent!;
            latent = _transformer.UnpatchifyLatent(tokens, hPacked, wPacked);
            tokens.Dispose();   // graph mode: the snapshot; eager: patchLatent itself (as before)
        }

        if (condCache is not null)
        {
            string uncondStats = uncondCache is not null
                ? $"; uncond {uncondCache.Computes} computes / {uncondCache.Reuses} reuses" : "";
            Logs.Info($"Step cache: cond {condCache.Computes} computes / {condCache.Reuses} reuses{uncondStats}");
            condCache.Dispose();
            uncondCache?.Dispose();
        }

        // condHidden/uncondHidden are owned by the prompt-embedding cache (host-materialized; survive across
        // gens for repeat prompts) — do NOT dispose here. They're released on cache eviction / pipeline Dispose.
        // extendedCond (region-extended) is rebuilt fresh every generation — always disposed.
        extendedCond?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        long phT5 = sw.ElapsedMilliseconds;
        Backend.Sync();
        DitShardBackend?.Sync();
        long phT6 = sw.ElapsedMilliseconds;
        if (streamer is not null)
        {
            // Streamed path always tears down regardless of KEEP_MODELS: nothing was fully resident to keep, and the
            // VAE decode needs the window's VRAM. Detach the hook first so a later eager generation on this same
            // transformer instance cannot call into a disposed controller.
            _transformer.BeforeBlockForward = null;
            streamer.EvictAll();
            streamer.Dispose();
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            _ditResident = false;
            // Unconditional, even though this generation never captured: a graph captured by an EARLIER resident
            // generation is still owned by this transformer at a matching signature, and streaming has since
            // re-uploaded every block to fresh addresses. The next resident generation would replay it against
            // freed memory (CUDA 700, context-poisoning). Idempotent and cheap.
            _transformer.InvalidateStepGraph(Backend);
        }
        else if (DitShardBackend is not null)
        {
            // Mirror of the sharding preload above: Backend frees shared + its block range, DitShardBackend frees
            // ONLY its block range. Freeing EnumerateWeights() on Backend alone (the unsharded call below) would
            // ask it to free tensors it never promoted (they're on DitShardBackend) and leave DitShardBackend's
            // range to accumulate across generations — a per-generation VRAM leak on the shard backend.
            if (!KeepModelsResident)
            {
                Backend.FreeWeights(_transformer.EnumerateSharedWeights());
                Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                DitShardBackend.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
                _ditResident = false;
            }
            // graphMode is always false when sharded (see its computation above), so no step-graph invalidation.
        }
        else if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
            // The captured step graph bakes weight device pointers — freeing the weights leaves it pointing at
            // freed memory (replay = CUDA 700 that poisons the context). Invalidate; next gen re-captures.
            if (graphMode)
                _transformer.InvalidateStepGraph(Backend);
        }

        long phT7 = sw.ElapsedMilliseconds;
        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        long phT8 = sw.ElapsedMilliseconds;
        Tensor image = _vaeDecoder.Decode(Backend, latent);
        long phT9 = sw.ElapsedMilliseconds;
        Logs.Verbose($"[krea2-phase] post-loop sync={phT6 - phT5}ms DiT-free={phT7 - phT6}ms VAE preload={phT8 - phT7}ms decode={phT9 - phT8}ms");
        latent.Dispose();

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        long phT10 = sw.ElapsedMilliseconds;
        // Device rgb conversion: CHW F32 → HWC u8 on-GPU, then one 3 MB D2H of the byte image (the host
        // TensorToRgbBytes path pulled 12 MB of float planes + ran a 12M-element CPU loop, ~140 ms).
        byte[] rgbData;
        {
            int outH = (int)image.Shape[2], outW = (int)image.Shape[3];
            Tensor hwcU8 = new Tensor(new TensorShape(outH, outW, 3), DType.U8);
            Backend.ChwF32ToHwcU8(hwcU8, image);
            rgbData = new byte[(long)outH * outW * 3];
            unsafe
            {
                fixed (byte* dst = rgbData)
                    Buffer.MemoryCopy((void*)hwcU8.DataPointer, dst, rgbData.Length, rgbData.Length);
            }
            hwcU8.Dispose();
        }
        Logs.Verbose($"[krea2-phase] rgb={sw.ElapsedMilliseconds - phT10}ms");
        image.Dispose();

        sw.Stop();
        Logs.Info($"Krea 2 {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgbData, width, height, seed);
    }

    /// <summary>Routes one patchified-space denoise step through <see cref="DitShardBackend"/>'s block-range split when configured, else the normal single-backend path. Sharding excludes step-cache (see <see cref="Krea2Transformer.ForwardPatchedSharded"/>) — <paramref name="stepCache"/> is only honored on the unsharded path; callers still pass it unconditionally, matching the existing call sites. <paramref name="attnBias"/> (regional prompting, Tier 3.7) is excluded from sharding the same way — <see cref="Krea2Transformer.ForwardPatchedSharded"/> has no bias parameter at all — so a non-null bias forces the unsharded path regardless of <see cref="DitShardBackend"/>; callers must log this once per generation (see <c>GenerateFromTokens</c>), not silently drop the conditioning.</summary>
    private Tensor RunForwardPatched(Tensor patchLatent, float t, Tensor encoderHidden, int hPacked, int wPacked,
        DeviceFeatureCache? stepCache, Tensor? attnBias = null)
    {
        if (DitShardBackend is not null && attnBias is null)
        {
            return _transformer.ForwardPatchedSharded(Backend, DitShardBackend, patchLatent, t, encoderHidden,
                hPacked, wPacked, DitShardSplitBlock);
        }
        return _transformer.ForwardPatched(Backend, patchLatent, t, encoderHidden, hPacked, wPacked, stepCache, attnBias);
    }

    /// <summary>Pixel-space counterpart of <see cref="RunForwardPatched"/> (img2img / masked-inpaint path).</summary>
    private Tensor RunForward(Tensor latent, float t, Tensor encoderHidden)
    {
        if (DitShardBackend is not null)
        {
            return _transformer.ForwardSharded(Backend, DitShardBackend, latent, t, encoderHidden, DitShardSplitBlock);
        }
        return _transformer.Forward(Backend, latent, t, encoderHidden);
    }

    /// <summary>Builds the initial latent. T2I: noise * initSigma. Img2img: the source is encoded through the Qwen-Image 3D-causal VAE encoder (already per-channel normalized to the transformer's latent space) and combined with fresh noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>.
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

    /// <summary>Streamed path only: returns the stream-ordered pool's reservations to the driver once per step.</summary>
    /// <remarks><c>retainBehind: 0</c> frees every block via <c>cuMemFreeAsync</c>, and <c>HARTSY_MEMPOOL_KEEP</c> (on
    /// by default) raises the pool's release threshold so those bytes stay reserved. That is a pure win when the DiT
    /// is resident (the same buffers get reused), but on the streamed path the pool grows by roughly a block per step.
    /// Measured on Ideogram 4: 4.1 → 11.6 GiB — the entire card — by step 17, versus a flat 5.1 GiB with this trim,
    /// for +1.4% wall-clock. See benchmarks/results/2026-07-27_lowvram_leak_fix.md §5f. Called from BOTH loop exits:
    /// the patchified fast path <c>continue</c>s before the loop body's tail.</remarks>
    private void TrimPoolOnStreamedPath(BlockStreamingController? streamer)
    {
        if (streamer is not null)
        {
            Backend.TrimMemoryPool();
        }
    }

    /// <summary>Per-forward activation/workspace estimate for one Krea 2 pass over the joint <c>[text, image]</c> sequence.</summary>
    /// <remarks>Single-stream: text and image share one sequence through all 28 blocks, so every term is counted once
    /// over the combined length at the block activation dtype (F16 on the default <c>HARTSY_DIT_F16</c> path). The
    /// SwiGLU term dominates — inner 16384 is 2.7× hidden. The flat 1 GB tail covers cuBLAS workspace, the fp8 GEMM
    /// dequant scratch and the RoPE tables. Deliberately generous: over-estimating costs a smaller prefetch window,
    /// under-estimating costs an OOM.</remarks>
    private static long EstimateActivationReserveBytes(int txtSeqLen, int imgSeqLen, Krea2Config config)
    {
        long seq = txtSeqLen + imgSeqLen;
        long actBytes = Models.Denoisers.DiTBlocks.DitDtype.Act == DType.F16 ? 2 : 4;
        long hidden = config.HiddenSize;
        long kvDim = (long)config.NumKvHeads * config.AttentionHeadDim;
        long ffnBytes = 3L * seq * config.IntermediateSize * actBytes;   // SwiGLU gate/up/silu/gated, ~3 live at once
        long qkvBytes = seq * (hidden * 3 + kvDim * 4) * actBytes;       // q/k/v, their permuted views, GQA-repeated k/v
        long attnBytes = 3L * seq * hidden * actBytes;                   // SDPA output, sigmoid gate, to_out
        long streamBytes = 6L * seq * hidden * actBytes;                 // residual + both norm/modulate pairs
        long f32Bytes = 2L * seq * hidden * 4;                           // F32 [text, image] concat + the final-layer tail
        return ffnBytes + qkvBytes + attnBytes + streamBytes + f32Bytes + 1024L * 1024 * 1024;
    }

    /// <summary>Releases the prompt-embedding cache (pipeline-internal state; see DiffusionPipelineBase).</summary>
    protected override void DisposeCore()
    {
        _cachedCond?.Dispose();
        _cachedCond = null;
        _cachedCondKey = null;
        _cachedUncond?.Dispose();
        _cachedUncond = null;
        _cachedUncondKey = null;
    }

    /// <summary>Encodes one region's prompt text (Tier 3.7) through the SAME tapped-layer encode the base prompt uses — the caller (recipe layer) must template + drop-index the region text identically to the base prompt (<c>Krea2RecipePipeline.EncodeWithTemplate</c>), since <see cref="EncodeTapped"/> has no template logic of its own.</summary>
    public Tensor EncodeRegionText(int[] tokenIds, int dropIndex) => EncodeTapped(tokenIds, dropIndex);

    /// <summary>Encodes a token sequence, stacks the 12 selected layers (tap-major <c>[1, S, 12·2560]</c>) and drops the first <paramref name="dropIndex"/> token positions (the chat-template system prefix).</summary>
    private unsafe Tensor EncodeTapped(int[] tokenIds, int dropIndex)
    {
        Tensor full = _textEncoder.EncodeMultiLayer(Backend, [tokenIds], Krea2Config.TextEncoderSelectLayers,
            interleavedLayout: false);
        if (dropIndex <= 0) return full;

        int s = (int)full.Shape[1];
        int feat = (int)full.Shape[2];
        int keep = Math.Max(1, s - dropIndex);
        Tensor sliced = new Tensor(new TensorShape(1, keep, feat), DType.F32);
        long rowBytes = (long)feat * sizeof(float);
        float* src = (float*)full.DataPointer + (long)dropIndex * feat;
        Buffer.MemoryCopy(src, (void*)sliced.DataPointer, (long)keep * rowBytes, (long)keep * rowBytes);
        full.Dispose();
        return sliced;
    }
}
