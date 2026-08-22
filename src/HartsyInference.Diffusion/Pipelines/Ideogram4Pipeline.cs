using HartsyInference.Core.MemoryManagement;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Ideogram 4 (<c>ideogram-oss/ideogram4</c>, 9.3B, non-commercial license) text-to-image pipeline. Orchestrates Qwen3-VL-8B (13-layer tap) → two single-stream DiTs (conditional + unconditional, asymmetric CFG) → Flux.2 VAE. Ported from upstream <c>pipeline_ideogram4.py</c>. Pipeline-level specifics:
/// <list type="bullet">
///   <item><b>Unified sequence</b> per prompt: <c>[text tokens][image tokens]</c>. Text features (Qwen 13-layer concat) sit at text positions; noise sits at image positions; an <c>image_indicator</c> embedding + 3D MRoPE keep them apart.</item>
///   <item><b>Asymmetric CFG</b>: the positive pass runs the conditional transformer over the full sequence and keeps only the image-token velocity; the negative pass runs the <i>unconditional</i> transformer over an image-only sequence with zeroed text features. Combined as <c>v = gw·pos + (1−gw)·neg</c> with a per-step guidance schedule (gw≈7 main, gw≈3 polish).</item>
///   <item><b>Logit-normal schedule</b> (<see cref="LogitNormalSchedule"/>), resolution-adjusted mean, plain Euler <c>z += v·(s−t)</c>.</item>
///   <item><b>Fixed-constant latent norm</b> (<see cref="Ideogram4LatentNorm"/>) applied to the packed token latent before the 2×2 unpatchify — NOT the Flux.2 VAE BatchNorm.</item>
/// </list>
/// <para><b>VRAM:</b> both 9.3B transformers must be resident during the loop (each step runs both), so this needs roughly 2× the DiT footprint plus the VAE. The Qwen encoder is freed before the loop. Realistically a multi-GPU / high-VRAM host; documented in the Phase 4 checklist.</para></summary>
public sealed unsafe class Ideogram4Pipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Ideogram4Transformer _conditional;
    private readonly Ideogram4Transformer _unconditional;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Ideogram4Config _config;

    private const int LlmTokenIndicator = 3;
    private const int OutputImageIndicator = 2;
    private const int ImagePositionOffset = 65536;

    /// <summary>Keeps BOTH 9.3B DiTs GPU-resident across generations (skips the post-loop FreeWeights + next-gen ~4.6 s re-upload). The TE cannot coexist with the resident DiTs (8 + 18.6 GB), so a prompt-cache MISS under this flag frees the DiTs first, encodes, then re-preloads — repeat prompts skip both. Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables) — the miss-path eviction above is what keeps smaller cards viable even with residency on.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    /// <summary>Calibrated step-cache ship point (HARTSY_STEP_CACHE=1): raw budget 0.3 confined to the LATE half of the schedule — 1.39× at SSIM 0.9530 on the 4090 A/B. No poly: Ideogram's block-0 indicator is schedule-flat while true residual drift falls 0.72→0.15, so a fitted map inverts the relationship (results doc 2026-07-22_accel_stepcache_ideogram_4090.md).</summary>
    private static readonly StepCacheProfile CalibratedStepCache =
        new(Threshold: 0.3f, Cap: 3, Poly: null, LateWindow: 0.5f);

    // Prompt-embedding cache (last prompt): the Qwen3-VL 13-layer tap keyed on the token ids. A hit skips the
    // whole TE phase (preload + encode + free ≈ 1.7 s). Reusing the SAME tensor reference also keeps the
    // conditional transformer's step-invariant llmProj cache warm across generations (keyed on this reference).
    private int[]? _cachedPromptKey;
    private Tensor? _cachedTextFeatures;

    // Position-id cache keyed on (numText, gridH, gridW): reusing the SAME tensor references keeps both
    // transformers' MRoPE cos/sin caches warm across generations.
    private (int NumText, int GridH, int GridW) _cachedPosKey = (-1, -1, -1);
    private Tensor? _cachedPosIds;
    private Tensor? _cachedPosIdsImageOnly;

    /// <summary>Creates an Ideogram 4 pipeline from pre-loaded components. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="textEncoder">Qwen3-VL-8B language tower (<see cref="LlamaStyleEncoderConfig.Qwen3_VL_8B"/>).</param>
    /// <param name="conditional">Conditional transformer (<c>transformer/</c> weights).</param>
    /// <param name="unconditional">Unconditional transformer (<c>unconditional_transformer/</c> weights).</param>
    /// <param name="vaeDecoder">Flux.2 VAE decoder (<c>VaeConfig.Flux2</c>).</param>
    /// <param name="config">Architecture config (pinned to the loaded transformers).</param>
    public Ideogram4Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Ideogram4Transformer conditional, Ideogram4Transformer unconditional,
        VaeDecoder vaeDecoder, Ideogram4Config config)
        : this(backend, textEncoder, conditional, unconditional, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates an Ideogram 4 pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>). Configure the encoder with <see cref="VaeConfig.Flux2"/> (its scalar scaling is identity; Ideogram's fixed-constant token norm is applied by the pipeline).</summary>
    public Ideogram4Pipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        Ideogram4Transformer conditional, Ideogram4Transformer unconditional,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, Ideogram4Config config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _conditional = conditional;
        _unconditional = unconditional;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Encodes arbitrary chat-templated text through this pipeline's own Qwen3-VL encoder — the identical multi-layer, interleaved tap-layer configuration (<see cref="Ideogram4Config.QwenActivationLayersHf"/>) the base prompt uses in <see cref="GenerateFromTokens"/>, so the feature dimension and layout match exactly. For regional/object prompt conditioning built by the caller (<see cref="Prompting.RegionalPromptResolver"/>'s <c>encodeRegion</c> delegate) — runs outside this method's own preload/free bracket around <see cref="_textEncoder"/>, so the caller pays a cold-weight touch if the encoder isn't already resident; correctness-only for this first pass, not a residency optimization. Returns a <c>[1, L, LlmFeaturesDim]</c> tensor; disposal is the caller's responsibility.</summary>
    public Tensor EncodeRegionText(int[] tokenIds) =>
        _textEncoder.EncodeMultiLayer(Backend, [tokenIds], Ideogram4Config.QwenActivationLayersHf, interleavedLayout: true);

    /// <summary>Generates an image from chat-templated prompt token ids (the Qwen3 chat template must already be applied). The negative branch needs no tokens — Ideogram's CFG zeroes the text features.
    /// <para>An <see cref="ImageToImageRequest"/> selects img2img: the source goes VAE-encode (32-ch latent) → 2×2
    /// token patchify (<c>[1, nImg, 128]</c>) → inverse fixed-constant latent norm → mix with fresh noise at the
    /// logit-normal time of the start step (Ideogram integrates <c>x_t = t·x0 + (1−t)·ε</c> from t≈0 noise to t≈1
    /// clean, so the mix uses <c>sigma = 1 − t(start)</c>) — requires a <see cref="VaeEncoder"/> on construction. A
    /// <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step packed blend in Ideogram's channel-inner
    /// token layout + final pixel recomposite). Strength=0 short-circuits to byte-identical pass-through.</para></summary>
    /// <param name="promptTokenIds">Tokenized, chat-templated prompt.</param>
    /// <param name="request">Width/Height/Seed (Steps and CfgScale come from <paramref name="preset"/>). Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint (strength maps onto the preset's step grid).</param>
    /// <param name="preset">Sampler preset (steps + guidance schedule + logit-normal mu/std). Defaults to <see cref="Ideogram4SamplerPreset.Default20"/>.</param>
    /// <param name="onProgress">Optional per-step callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        TextToImageRequest request,
        Ideogram4SamplerPreset? preset = null,
        Action<GenerationProgress>? onProgress = null,
        RegionalPlan? regionalPlan = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        preset ??= Ideogram4SamplerPreset.Default20;
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        if (promptTokenIds.Length == 0)
            throw new ArgumentException("Prompt token ids must be non-empty.", nameof(promptTokenIds));
        if (promptTokenIds.Length > _config.MaxTextTokens)
            throw new ArgumentException($"Prompt has {promptTokenIds.Length} tokens, exceeds MaxTextTokens={_config.MaxTextTokens}.", nameof(promptTokenIds));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Generic.Width;
        int height = request.Height ?? GenerationDefaults.Generic.Height;
        int patch = _config.PatchSize * _config.VaeScaleFactor; // 2 × 8 = 16
        if (width % patch != 0 || height % patch != 0)
            throw new ArgumentException($"Width and height must be divisible by {patch} for Ideogram 4.");

        int gridH = height / patch;
        int gridW = width / patch;
        int numImageTokens = gridH * gridW;
        int numText = promptTokenIds.Length;
        int seqLen = numText + numImageTokens;
        int latentDim = _config.InChannels;
        int steps = preset.NumSteps;

        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;
        // Ideogram's loop runs i = steps−1 → 0 (t: ≈0 noise → ≈1 clean); skipping `startStep` iterations from the
        // noisy end means starting at loop index steps−1−startStep, with the latent seeded at t(startIdx+1).
        int startIdx = steps - 1 - startStep;

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "t2i";
        Logs.Info($"Ideogram 4 {opMode}: {width}x{height}, preset={preset.Name} ({steps} steps), seed={seed}");
        Logs.Warning("Ideogram 4 runs TWO 9.3B transformers concurrently (asymmetric CFG) — expect very high VRAM use.");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompt: Qwen3-VL 13-layer concat → [1, numText, llmFeaturesDim] ──
        // Diagnostic: log the token ids so they can be compared 1:1 against a reference
        // transformers Qwen3-VL run (tests/python-reference/dump_qwen3vl_ideogram4_encoder.py).
        // A token-id mismatch ⇒ tokenizer/chat-template bug; a match with differing encoder
        // features ⇒ encoder forward/weight-load bug.
        LogTokenIds(promptTokenIds);
        bool promptHit = _cachedTextFeatures is not null && _cachedPromptKey is not null
            && _cachedPromptKey.AsSpan().SequenceEqual(promptTokenIds);
        Tensor textFeatures;
        if (promptHit)
        {
            textFeatures = _cachedTextFeatures!;
            Logs.Info("[Ideogram4] prompt-embedding cache hit — TE phase skipped");
        }
        else
        {
            if (_ditResident)
            {
                // The 8 GB TE cannot coexist with the 18.6 GB of resident DiTs (HARTSY_KEEP_MODELS); evict
                // them for this new-prompt generation and re-preload below.
                Backend.Sync();
                Backend.FreeWeights(_conditional.EnumerateWeights());
                Backend.FreeWeights(_unconditional.EnumerateWeights());
                _ditResident = false;
            }
            Backend.PreloadWeights(_textEncoder.EnumerateWeights());
            // interleavedLayout: true — Ideogram 4 concatenates the 13 taps HIDDEN-MAJOR (c = h*13 + tap), matching
            // upstream pipeline_ideogram4.py (permute(stack,(1,2,3,0)).reshape) and ComfyUI (permute(0,2,3,1)). The
            // llm_cond_norm/llm_cond_proj weights are trained on this order; tap-major scrambles every input channel.
            textFeatures = _textEncoder.EncodeMultiLayer(Backend, [promptTokenIds], Ideogram4Config.QwenActivationLayersHf, interleavedLayout: true);
            if ((int)textFeatures.Shape[2] != _config.LlmFeaturesDim)
                throw new InvalidOperationException($"Encoder produced {textFeatures.Shape[2]}-dim features, expected {_config.LlmFeaturesDim} (13 × 4096). Check the tap-layer indices / encoder preset.");
            Backend.Sync();
            LogEncoderStats(textFeatures);
            // When IDEOGRAM4_DEBUG_DIR is set, dump the raw encoder output so it can be diffed
            // element-wise against the transformers Qwen3-VL reference (same tokens).
            HartsyInference.Diffusion.Models.Denoisers.Ideogram4DebugDump.Dump("textFeatures", textFeatures);
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
            // Materialize the encoder output on the host, then reclaim every encoder intermediate. The Qwen3-VL
            // forward leaves hundreds of device-cached activations that otherwise linger until GC finalization and
            // hold multi-GB into the DiT phase. textFeatures is already host-synced (LogEncoderStats read it), but
            // touch it explicitly — FreeActivations frees device buffers WITHOUT a D2H sync-back, so anything
            // still device-only would be silently lost.
            _ = textFeatures.DataPointer;
            Backend.FreeActivations();
            _cachedTextFeatures?.Dispose();
            _cachedTextFeatures = textFeatures;
            _cachedPromptKey = (int[])promptTokenIds.Clone();
        }
        Logs.Info($"Prompt encoded in {sw.ElapsedMilliseconds}ms");

        // ── 2. Build the unified-sequence conditioning tensors ──
        // Regional conditioning appends each region's encoded features after the base text tokens
        // so image tokens can be biased toward their region's prompt (RegionalAttentionBias). With
        // no plan this collapses exactly to the base single-prompt path.
        bool hasRegions = regionalPlan is not null && regionalPlan.Regions.Count > 0;
        int effNumText = numText;
        List<(int Start, int End)>? regionRanges = null;
        List<float[]>? regionGridMasks = null;
        float[]? regionWeights = null;
        if (hasRegions)
        {
            (effNumText, regionRanges, regionGridMasks) = BuildRegionLayout(regionalPlan!, numText, numImageTokens, gridH, gridW);
            regionWeights = new float[regionalPlan!.Regions.Count];
        }
        int effSeqLen = effNumText + numImageTokens;
        // Text rows only ([1, effNumText, llmDim]) — the transformer synthesizes the zero image rows and caches
        // the projection keyed on this reference, so pass the cached tensor itself when there are no regions.
        Tensor llmTextRows = hasRegions
            ? BuildLlmFull(textFeatures, numText, effNumText, _config.LlmFeaturesDim, regionalPlan)
            : textFeatures;
        (Tensor posIds, Tensor posIdsImageOnly) = GetOrBuildPositionIds(effNumText, gridH, gridW);
        int[] indicator = BuildIndicator(effNumText, numImageTokens);         // [L]
        int[] indicatorImageOnly = new int[numImageTokens];
        Array.Fill(indicatorImageOnly, OutputImageIndicator);

        // ── 3. Schedule ──
        // Sampler selection is NOT wired on this family (2026-08-20 audit), and three independent things block it:
        // (1) there is no FlowMatchEulerDiscreteScheduler here at all — the schedule is a LogitNormalSchedule plus a
        // step-interval grid, so the sampler core has no sigma array to integrate over; (2) the denoise loop runs
        // BACKWARD (startIdx down to 0) while ISampler indexes its sigmas forward; (3) guidance comes from the
        // preset's own per-step schedule (gw ~7 main, ~3 polish) keyed on the descending index. Converting means
        // restructuring the loop, not swapping a call. Refuse a named sampler rather than accepting it and sampling
        // with something else.
        if (!string.IsNullOrWhiteSpace(request.Scheduler))
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' is not available on Ideogram 4. This family samples on its "
                + "own logit-normal schedule with a per-step guidance preset and has not been converted to the "
                + "sampler seam. Leave the sampler unset.");
        }
        LogitNormalSchedule schedule = LogitNormalSchedule.ForResolution(height, width, preset.Mu, preset.Std);
        float[] grid = LogitNormalSchedule.MakeStepIntervals(steps);

        // ── 3b. Initial latent (image-token format) [1, nImg, 128] — t2i: pure noise; img2img: encoded source
        //        mixed with noise at the start step's logit-normal time. Masked inpaint keeps the clean token
        //        source + packed mask for the per-step blend.
        (Tensor z, Tensor? sourceTokens) = BuildInitialTokenLatent(
            request, schedule, grid, numImageTokens, latentDim, gridH, gridW, seed, startIdx,
            keepSourceLatent: isMaskedInpaint);
        Tensor? packedMask = null;
        if (isMaskedInpaint)
        {
            Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(
                maskPixel!, gridH * _config.PatchSize, gridW * _config.PatchSize);
            packedMask = MaskBlendUtilities.PackLatentMask2x2(
                latentMask, gridH * _config.PatchSize, gridW * _config.PatchSize);
            latentMask.Dispose();
        }

        // ── 4b. Optional across-step First-Block cache (HARTSY_STEP_CACHE, default off — reference wiring
        // is QwenImagePipeline / INFERENCE_ACCEL_GRIND §H1.4). One instance per transformer: the conditional
        // and unconditional models are DIFFERENT 9.3B weight sets, so their hidden states never mix. Regional
        // plans are excluded — the per-step attention bias changes the block math under the residual's feet.
        (float stepCacheThreshold, int stepCacheCap, float[]? stepCachePoly, float stepCacheLate) =
            StepCacheEnv.Resolve(CalibratedStepCache);
        DeviceFeatureCache? condCache = null;
        DeviceFeatureCache? uncondCache = null;
        if (stepCacheThreshold > 0f && !hasRegions)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                condCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                uncondCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}"
                    + (stepCacheLate > 0f ? $", lateWindow={stepCacheLate}" : ""));
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }

        // ── 5. Denoise loop ──
        // Ideogram 4 is the worst case for a small card: asymmetric CFG runs a SEPARATE 9.3 GB transformer for the
        // conditional and unconditional passes, and both were held resident for the whole loop (~18.6 GB) — which is
        // what the recipe's old hardcoded ">= 20 GB free VRAM" refusal was compensating for.
        //
        // Streaming both is the fix, and it is a better one than alternating whole transformers: the two are used
        // back-to-back within EVERY step, so a whole-DiT swap would move 18.6 GB per step, whereas two independent
        // sliding windows move one block at a time and keep only (prefetch+1) blocks of each resident.
        // Each transformer gets its OWN controller — they have distinct weights and interleave within a step, so a
        // shared window would thrash.
        BlockStreamingController? condStreamer = null;
        BlockStreamingController? uncondStreamer = null;
        if (Backend.StreamingCache is not null)
        {
            long condBytes = SumBlockBytes(_conditional);
            long uncondBytes = SumBlockBytes(_unconditional);
            long sharedBytes = SumBytes(_conditional.EnumerateSharedWeights()) + SumBytes(_unconditional.EnumerateSharedWeights());
            // The shared sets stay resident on both paths, so they are part of what the block budget must fit beside.
            long reserve = EstimateActivationReserveBytes(seqLen, _config.EmbDim, _config.IntermediateSize) + sharedBytes;

            VramPlanner planner = new VramPlanner(Backend.StreamingCache, "Ideogram4", Backend);
            PhasePlacement placement = planner.PlanPhase(
                "denoise", condBytes + uncondBytes, reserve, alreadyResident: _ditResident, canStream: true);
            if (placement == PhasePlacement.Resident)
            {
                if (!_ditResident)
                {
                    Backend.PreloadWeights(_conditional.EnumerateWeights());
                    Backend.PreloadWeights(_unconditional.EnumerateWeights());
                }
                _ditResident = true;
            }
            else
            {
                Backend.PreloadWeights(_conditional.EnumerateSharedWeights());
                Backend.PreloadWeights(_unconditional.EnumerateSharedWeights());
                condStreamer = AttachStreamer(_conditional, reserve);
                uncondStreamer = AttachStreamer(_unconditional, reserve);
                _ditResident = false;
                Logs.Info($"Ideogram4 streaming: 2 x {_conditional.BlockCount} blocks, " +
                    $"cond ~{condBytes / (1024 * 1024)} MB + uncond ~{uncondBytes / (1024 * 1024)} MB, " +
                    $"shared ~{sharedBytes / (1024 * 1024)} MB resident, reserve ~{reserve / (1024 * 1024)} MB");
            }
        }
        else
        {
            Backend.PreloadWeights(_conditional.EnumerateWeights());
            Backend.PreloadWeights(_unconditional.EnumerateWeights());
            _ditResident = true;
        }
        LogVram("after DiT weight preload");

        // Materialize everything that must survive across steps, then reclaim conditioning-build leftovers.
        // All of these are host-created (BuildLlmFull / BuildPositionIds / CreateNoise write on the CPU), so the
        // touches are free — they just guarantee nothing here is a live GPU activation when the per-step
        // FreeActivations (which frees device buffers WITHOUT a D2H sync-back) runs below.
        _ = llmTextRows.DataPointer;
        _ = posIds.DataPointer;
        _ = posIdsImageOnly.DataPointer;
        _ = z.DataPointer;
        if (sourceTokens is not null) _ = sourceTokens.DataPointer;
        if (packedMask is not null) _ = packedMask.DataPointer;
        Backend.FreeActivations();

        for (int i = startIdx; i >= 0; i--)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            Backend.ResetD2hSyncCount();
            HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Reset();
            float tVal = schedule.Map(grid[i + 1]);
            float sVal = schedule.Map(grid[i]);
            float delta = sVal - tVal;
            float gw = preset.GuidanceSchedule[i];

            // Positive pass: full sequence, image-token velocity returned directly. Regional bias (when
            // present) steers each image token toward its region's appended conditioning columns.
            Tensor? regionBias = null;
            if (hasRegions)
            {
                regionalPlan!.ResolveStep(steps - 1 - i, regionWeights!);
                regionBias = RegionalAttentionBias.Build(effSeqLen, effNumText, numImageTokens, regionRanges!, regionGridMasks!, regionWeights!);
            }
            // Late-window gate (HARTSY_STEP_CACHE_LATE): pass the caches only inside the last `late` fraction
            // of the schedule — Ideogram's residual drift falls ~5× from early to late steps while its
            // block-0 indicator stays flat, so early reuse is where the quality damage concentrates. Outside
            // the window the forward is byte-identical uncached (first armed step recomputes and snapshots).
            bool cacheEligible = stepCacheLate <= 0f || (steps - i) > steps * (1f - stepCacheLate);
            Tensor posV = _conditional.Forward(Backend, llmTextRows, z, tVal, posIds, indicator, regionBias,
                cacheEligible ? condCache : null);
            regionBias?.Dispose();

            // Negative pass: image-only sequence, unconditional transformer. Null text features — the zeroed
            // text projection is exactly zero, so the transformer skips the text path outright.
            Tensor negV = _unconditional.Forward(Backend, null, z, tVal, posIdsImageOnly, indicatorImageOnly, null,
                cacheEligible ? uncondCache : null);

            // Conditioning-effect probe (profile-gated): relative RMS difference between the conditional
            // and unconditional velocities. ~0 means the prompt has no effect (encoder/feature bug);
            // a substantial value means text conditioning is live. Reads DataPointer (2 D2H syncs), so
            // it only runs under HARTSY_DIT_PROFILE=1 to keep real runs fully resident.
            double cfgDelta = -1;
            if (HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Enabled)
            {
                float* pp = (float*)posV.DataPointer;
                float* np = (float*)negV.DataPointer;
                long nv = posV.ElementCount;
                double sd = 0, sn = 0;
                for (long e = 0; e < nv; e++) { double d = pp[e] - np[e]; sd += d * d; sn += (double)np[e] * np[e]; }
                cfgDelta = Math.Sqrt(sd / nv) / (Math.Sqrt(sn / nv) + 1e-9);
            }

            // v = gw·pos + (1−gw)·neg ; z = z + v·delta — in-place on the GPU-resident latent.
            Backend.CfgEulerStep(z, posV, negV, gw, delta);
            posV.Dispose();
            negV.Dispose();

            // Masked-inpaint blend: keep the unmasked region on the source's trajectory by re-mixing the source
            // with fresh noise at the step's new time sVal (sigma = 1 − t in Ideogram's t≈0-noise → t≈1-clean
            // convention). Final step (i=0) blends with the clean source. Host-side: reading z.DataPointer syncs
            // the GPU latent back; the per-step FreeActivations below drops the stale device copy so the next
            // forward re-uploads the blended host data.
            if (packedMask is not null && sourceTokens is not null)
            {
                Tensor noisedSource;
                if (i > 0)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(z.Shape, seed + i);
                    noisedSource = new Tensor(z.Shape, DType.F32);
                    Img2ImgSetup.MixAtSigma(noisedSource, sourceTokens, freshNoise, 1.0f - sVal);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourceTokens;
                }
                MaskBlendUtilities.BlendPackedChannelInnerInPlace(z, noisedSource, packedMask);
                if (noisedSource != sourceTokens) noisedSource.Dispose();
            }

            stepSw.Stop();
            (long freeB, long totalB) = Backend.GetVramInfo();
            long syncs = Backend.GetD2hSyncCount();
            string vram = totalB > 0 ? $"{(totalB - freeB) / 1073741824.0:F1}/{totalB / 1073741824.0:F1} GiB used" : "n/a";
            string profile = HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.Enabled
                ? $" | attn {HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.AttentionMs:F0}ms mlp {HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.Ideogram4Profile.MlpMs:F0}ms (both passes) | cfgΔ {cfgDelta:F3}"
                : "";
            Logs.Info($"[Ideogram4] step {steps - i}/{steps} t={tVal:F3} gw={gw:F1} {stepSw.ElapsedMilliseconds}ms | VRAM {vram} | D2H syncs {syncs}{profile}");
            // Tag the latent family for live-preview decoders (Flux.2 VAE, shared with Lens). The working
            // latent is token-packed [1, nImg, 128], so no per-step Latent snapshot is provided.
            onProgress?.Invoke(new GenerationProgress(steps - i, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                LatentArch = LatentArchitecture.Flux2,
            });

            // Masked inpaint is the one path that must read z on the host every step (the blend above already
            // did); protect the blended host copy from the next forward's stale device pointer with a per-step
            // FreeActivations. The plain loop stays DRAIN-FREE: every intermediate is explicitly disposed, z
            // stays device-resident across steps (CfgEulerStep is in-place), and the post-log z.DataPointer
            // materialize + FreeActivations that used to run here cost a measured ~325 ms/step of pipeline
            // drain (step wall 1156 ms vs 830 ms issued).
            if (packedMask is not null)
            {
                Backend.FreeActivations();
            }

            // Streamed path only, and exactly ONCE per step even though two windows are live: the mempool is
            // device-global, so a second trim in the same step would sync the stream again and find nothing.
            // Rationale and measurements live on BlockStreamingController.TrimAfterStep.
            (condStreamer ?? uncondStreamer)?.TrimAfterStep();
        }

        if (condCache is not null)
        {
            Logs.Info($"Step cache: cond {condCache.Computes} computes / {condCache.Reuses} reuses; " +
                $"uncond {uncondCache!.Computes} computes / {uncondCache.Reuses} reuses");
            condCache.Dispose();
            uncondCache.Dispose();
        }

        // llmTextRows/posIds/posIdsImageOnly are cross-generation caches; only a per-gen regional concat is ours.
        if (hasRegions) llmTextRows.Dispose();
        sourceTokens?.Dispose();
        packedMask?.Dispose();

        // ── 6. Free DiT weights before VAE decode (skipped under HARTSY_KEEP_MODELS — the Flux.2 VAE decode
        // falls back to tiled if the resident DiTs leave too little room for the full-res im2col) ──
        Backend.Sync();
        if (condStreamer is not null || uncondStreamer is not null)
        {
            // Streamed path always tears down, regardless of KEEP_MODELS: nothing was fully resident to keep, and the
            // VAE decode needs the block window's VRAM back. Detach the hooks first so a later eager generation on
            // these same transformer instances does not call into a disposed controller.
            _conditional.BeforeBlockForward = null;
            _unconditional.BeforeBlockForward = null;
            condStreamer?.EvictAll();
            condStreamer?.Dispose();
            uncondStreamer?.EvictAll();
            uncondStreamer?.Dispose();
            Backend.FreeWeights(_conditional.EnumerateSharedWeights());
            Backend.FreeWeights(_unconditional.EnumerateSharedWeights());
            _ditResident = false;
        }
        else if (!KeepModelsResident)
        {
            Backend.FreeWeights(_conditional.EnumerateWeights());
            Backend.FreeWeights(_unconditional.EnumerateWeights());
            _ditResident = false;
        }

        // ── 7. Latent un-normalize (fixed constants) + unpatchify → [1, 32, H/8, W/8] ──
        Stopwatch tailSw = Stopwatch.StartNew();
        ApplyLatentNorm(z);
        Tensor vaeIn = Unpatchify(z, gridH, gridW, _config.PatchSize);
        z.Dispose();
        Logs.Verbose($"[Ideogram4] latent-norm + unpatchify in {tailSw.ElapsedMilliseconds}ms");

        // ── 8. VAE decode ──
        Logs.Verbose("Decoding latents (Flux.2 VAE, tiled)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, vaeIn);
        Backend.Sync();
        vaeSw.Stop();
        Logs.Info($"[Ideogram4] VAE decode in {vaeSw.ElapsedMilliseconds}ms");
        vaeIn.Dispose();

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        tailSw.Restart();
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();
        Logs.Verbose($"[Ideogram4] rgb conversion in {tailSw.ElapsedMilliseconds}ms");

        // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the budget of whatever generation runs next.
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Ideogram 4 {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, width, height, seed);
    }

    /// <summary>Builds the initial image-token latent <c>[1, nImg, 128]</c>. T2I: pure seeded noise (Ideogram's integrator starts at t≈0). Img2img: the source goes VAE-encode (<c>[1, 32, H/8, W/8]</c>, Flux2 scaling is identity) → 2×2 token patchify (inverse of <see cref="Unpatchify"/>) → inverse fixed-constant latent norm (<c>(x − Shift)/Scale</c>, inverse of <see cref="ApplyLatentNorm"/>) → <c>Img2ImgSetup.MixAtSigma</c> with fresh noise at <c>sigma = 1 − t(startIdx+1)</c> (Ideogram's <c>x_t = t·x0 + (1−t)·ε</c>).
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the clean normalized source tokens
    /// are returned alongside for the per-step blend. Caller disposes both. Source is null for t2i and plain
    /// img2img.</para></summary>
    private (Tensor z, Tensor? sourceTokens) BuildInitialTokenLatent(
        TextToImageRequest request, LogitNormalSchedule schedule, float[] grid,
        int numImageTokens, int latentDim, int gridH, int gridW, int seed, int startIdx,
        bool keepSourceLatent)
    {
        TensorShape tokenShape = new TensorShape(1, numImageTokens, latentDim);
        Tensor noise = TakeOrCreateNoise(request, tokenShape, seed);
        if (request is not ImageToImageRequest img2img) return (noise, null);

        Stopwatch vaeEncSw = Stopwatch.StartNew();
        Tensor sourceVae = _vaeEncoder!.Encode(Backend, img2img.SourceImage);   // [1, 32, H/8, W/8]
        vaeEncSw.Stop();
        Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

        Tensor sourceTokens = PatchifyToTokens(sourceVae, gridH, gridW, _config.PatchSize);   // [1, nImg, 128]
        sourceVae.Dispose();
        ApplyInverseLatentNorm(sourceTokens);

        Tensor z = new Tensor(tokenShape, DType.F32);
        float tStart = schedule.Map(grid[startIdx + 1]);
        Img2ImgSetup.MixAtSigma(z, sourceTokens, noise, 1.0f - tStart);
        noise.Dispose();
        if (keepSourceLatent)
        {
            return (z, sourceTokens);
        }
        sourceTokens.Dispose();
        return (z, null);
    }

    /// <summary>Inverse of <see cref="ApplyLatentNorm"/>, in-place: <c>z[...,c] = (z[...,c] − Shift[c]) / Scale[c]</c> — maps a VAE-space packed latent into the token space the transformers denoise.</summary>
    private static void ApplyInverseLatentNorm(Tensor z)
    {
        int channels = (int)z.Shape[2];
        if (channels != Ideogram4LatentNorm.Channels)
            throw new InvalidOperationException($"Latent channels {channels} != {Ideogram4LatentNorm.Channels}.");
        long tokens = z.Shape[0] * z.Shape[1];
        float* zp = (float*)z.DataPointer;
        float[] scale = Ideogram4LatentNorm.Scale;
        float[] shift = Ideogram4LatentNorm.Shift;
        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOff = tok * channels;
            for (int c = 0; c < channels; c++)
                zp[baseOff + c] = (zp[baseOff + c] - shift[c]) / scale[c];
        }
    }

    /// <summary>Inverse of <see cref="Unpatchify"/>: VAE latent <c>[1, aeC, gridH·patch, gridW·patch]</c> → packed token latent <c>[1, gridH·gridW, patch²·aeC]</c> with feature <c>f = p1·(aeC·patch) + p2·aeC + ae</c>.</summary>
    private static Tensor PatchifyToTokens(Tensor vaeLatent, int gridH, int gridW, int patch)
    {
        int aeC = (int)vaeLatent.Shape[1];
        int inH = (int)vaeLatent.Shape[2];
        int inW = (int)vaeLatent.Shape[3];
        if (inH != gridH * patch || inW != gridW * patch)
            throw new ArgumentException(
                $"VAE latent {inH}x{inW} does not match token grid {gridH}x{gridW} at patch {patch}.", nameof(vaeLatent));
        int packedC = aeC * patch * patch;
        Tensor tokens = new Tensor(new TensorShape(1, (long)gridH * gridW, packedC), DType.F32);
        float* src = (float*)vaeLatent.DataPointer;
        float* dst = (float*)tokens.DataPointer;
        for (int gh = 0; gh < gridH; gh++)
        {
            for (int gw = 0; gw < gridW; gw++)
            {
                long token = (long)gh * gridW + gw;
                long dstBase = token * packedC;
                for (int p1 = 0; p1 < patch; p1++)
                {
                    for (int p2 = 0; p2 < patch; p2++)
                    {
                        int srcH = gh * patch + p1;
                        int srcW = gw * patch + p2;
                        for (int ae = 0; ae < aeC; ae++)
                        {
                            int f = p1 * (aeC * patch) + p2 * aeC + ae;
                            long srcOff = ((long)ae * inH + srcH) * inW + srcW;
                            dst[dstBase + f] = src[srcOff];
                        }
                    }
                }
            }
        }
        return tokens;
    }

    /// <summary>Places encoded base text features <c>[1, numText, D]</c> (and any regional features) into a <c>[1, seqLen, D]</c> tensor: base text at the front, region features immediately after, zeros in any remainder.</summary>
    private static Tensor BuildLlmFull(Tensor textFeatures, int numText, int seqLen, int dim, RegionalPlan? plan)
    {
        Tensor full = new Tensor(new TensorShape(1, seqLen, dim), DType.F32);
        float* dst = (float*)full.DataPointer;
        new Span<float>(dst, checked((int)((long)seqLen * dim))).Clear();
        float* src = (float*)textFeatures.DataPointer;
        long baseBytes = (long)numText * dim * sizeof(float);
        Buffer.MemoryCopy(src, dst, baseBytes, baseBytes);
        if (plan is not null)
        {
            long offsetTokens = numText;
            foreach (RegionConditioning region in plan.Regions)
            {
                int len = (int)region.Cond.Shape[1];
                float* regionSrc = (float*)region.Cond.DataPointer;
                long regionBytes = (long)len * dim * sizeof(float);
                Buffer.MemoryCopy(regionSrc, dst + offsetTokens * dim, regionBytes, regionBytes);
                offsetTokens += len;
            }
        }
        return full;
    }

    /// <summary>Computes the regional layout for the unified sequence: the extended text-token count, each region's conditioning column range, and each region's image-grid mask (row-major <c>[numImg]</c>).</summary>
    private (int ExtNumText, List<(int Start, int End)> Ranges, List<float[]> GridMasks) BuildRegionLayout(
        RegionalPlan plan, int numText, int numImg, int gridH, int gridW)
    {
        List<(int Start, int End)> ranges = new List<(int Start, int End)>(plan.Regions.Count);
        List<float[]> masks = new List<float[]>(plan.Regions.Count);
        int cursor = numText;
        foreach (RegionConditioning region in plan.Regions)
        {
            if ((int)region.Cond.Shape[2] != _config.LlmFeaturesDim)
            {
                throw new InvalidOperationException($"Region conditioning dim {region.Cond.Shape[2]} != LlmFeaturesDim {_config.LlmFeaturesDim}.");
            }
            int len = (int)region.Cond.Shape[1];
            ranges.Add((cursor, cursor + len));
            cursor += len;
            Tensor latentMask = region.Mask.ToLatentMask(gridH, gridW);
            float[] grid = latentMask.AsReadOnlySpan<float>().ToArray();
            latentMask.Dispose();
            if (grid.Length != numImg)
            {
                throw new InvalidOperationException($"Region grid mask length {grid.Length} != numImg {numImg}.");
            }
            masks.Add(grid);
        }
        return (cursor, ranges, masks);
    }

    /// <summary>Builds <c>[1, L, 3]</c> MRoPE positions: text tokens <c>(i, i, i)</c>; image tokens <c>(0, row, col) + IMAGE_POSITION_OFFSET</c>.</summary>
    private static Tensor BuildPositionIds(int numText, int gridH, int gridW)
    {
        int numImg = gridH * gridW;
        int seqLen = numText + numImg;
        Tensor pos = new Tensor(new TensorShape(1, seqLen, 3), DType.F32);
        float* p = (float*)pos.DataPointer;
        for (int t = 0; t < numText; t++)
        {
            long off = (long)t * 3;
            p[off + 0] = t;
            p[off + 1] = t;
            p[off + 2] = t;
        }
        for (int idx = 0; idx < numImg; idx++)
        {
            int row = idx / gridW;
            int col = idx % gridW;
            long off = (long)(numText + idx) * 3;
            p[off + 0] = ImagePositionOffset;
            p[off + 1] = ImagePositionOffset + row;
            p[off + 2] = ImagePositionOffset + col;
        }
        return pos;
    }

    /// <summary>Slices the image-token rows <c>[numText..L)</c> out of a <c>[1, L, 3]</c> position tensor into <c>[1, numImg, 3]</c>.</summary>
    private static Tensor SliceImagePositions(Tensor posIds, int numText, int numImg)
    {
        Tensor outP = new Tensor(new TensorShape(1, numImg, 3), DType.F32);
        float* src = (float*)posIds.DataPointer;
        float* dst = (float*)outP.DataPointer;
        long bytes = (long)numImg * 3 * sizeof(float);
        Buffer.MemoryCopy(src + (long)numText * 3, dst, bytes, bytes);
        return outP;
    }

    private static int[] BuildIndicator(int numText, int numImg)
    {
        int[] ind = new int[numText + numImg];
        for (int t = 0; t < numText; t++) ind[t] = LlmTokenIndicator;
        for (int i = 0; i < numImg; i++) ind[numText + i] = OutputImageIndicator;
        return ind;
    }

    /// <summary>Builds (or returns the cached) MRoPE position tensors for the given layout. Reusing the same tensor references across generations keeps both transformers' cos/sin caches warm.</summary>
    private (Tensor PosIds, Tensor PosIdsImageOnly) GetOrBuildPositionIds(int numText, int gridH, int gridW)
    {
        if (_cachedPosIds is not null && _cachedPosIdsImageOnly is not null
            && _cachedPosKey == (numText, gridH, gridW))
        {
            return (_cachedPosIds, _cachedPosIdsImageOnly);
        }
        _cachedPosIds?.Dispose();
        _cachedPosIdsImageOnly?.Dispose();
        _cachedPosIds = BuildPositionIds(numText, gridH, gridW);
        _cachedPosIdsImageOnly = SliceImagePositions(_cachedPosIds, numText, gridH * gridW);
        _cachedPosKey = (numText, gridH, gridW);
        return (_cachedPosIds, _cachedPosIdsImageOnly);
    }

    /// <summary>Diagnostic: logs the prompt token ids (count + a head/tail window) so they can be compared 1:1 against a reference transformers Qwen3-VL tokenization of the same prompt + chat template.</summary>
    private static void LogTokenIds(int[] ids)
    {
        int head = Math.Min(24, ids.Length);
        string headStr = string.Join(",", ids[..head]);
        string tailStr = ids.Length > 24 ? " … " + string.Join(",", ids[^Math.Min(8, ids.Length)..]) : "";
        Logs.Info($"[Ideogram4] promptTokens count={ids.Length} ids=[{headStr}{tailStr}]");
    }

    /// <summary>Diagnostic: logs summary stats of the Qwen3-VL encoder output. The text features are already host-synced (BuildLlmFull reads them on CPU), so this is nearly free. Run two DIFFERENT prompts and compare: identical stats ⇒ the encoder is producing prompt-independent features (weight-load / tokenization bug); differing stats but an image that still ignores the prompt ⇒ the bug is downstream in the DiT conditioning or MRoPE. NaN/inf &gt; 0 ⇒ the encoder forward is numerically broken.</summary>
    private static void LogEncoderStats(Tensor textFeatures)
    {
        float* tf = (float*)textFeatures.DataPointer;
        long n = textFeatures.ElementCount;
        if (n == 0) return;
        double sum = 0, sumsq = 0;
        float mn = float.MaxValue, mx = float.MinValue;
        long badCount = 0;
        for (long e = 0; e < n; e++)
        {
            float v = tf[e];
            if (float.IsNaN(v) || float.IsInfinity(v)) { badCount++; continue; }
            sum += v; sumsq += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        long good = n - badCount;
        double mean = good > 0 ? sum / good : 0;
        double std = good > 0 ? Math.Sqrt(Math.Max(0, sumsq / good - mean * mean)) : 0;
        Logs.Info($"[Ideogram4] textFeatures [{textFeatures.Shape[1]}x{textFeatures.Shape[2]}] " +
            $"mean={mean:F5} std={std:F5} min={mn:F4} max={mx:F4} nan/inf={badCount} " +
            $"first=[{tf[0]:F4}, {tf[1]:F4}, {tf[2]:F4}, {tf[3]:F4}]");
    }

    /// <summary>Logs current device VRAM usage at an Info-level checkpoint (no-op detail on the CPU backend).</summary>
    private void LogVram(string stage)
    {
        (long freeB, long totalB) = Backend.GetVramInfo();
        if (totalB > 0)
            Logs.Info($"[Ideogram4] VRAM {stage}: {(totalB - freeB) / 1073741824.0:F1}/{totalB / 1073741824.0:F1} GiB used ({freeB / 1073741824.0:F1} free)");
    }

    /// <summary>Applies the fixed per-channel latent norm in-place: <c>z[...,c] = z[...,c]·Scale[c] + Shift[c]</c> on a <c>[1, nImg, 128]</c> packed latent.</summary>
    private void ApplyLatentNorm(Tensor z)
    {
        int channels = (int)z.Shape[2];
        if (channels != Ideogram4LatentNorm.Channels)
            throw new InvalidOperationException($"Latent channels {channels} != {Ideogram4LatentNorm.Channels}.");
        long tokens = z.Shape[0] * z.Shape[1];
        float* zp = (float*)z.DataPointer;
        float[] scale = Ideogram4LatentNorm.Scale;
        float[] shift = Ideogram4LatentNorm.Shift;
        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOff = tok * channels;
            for (int c = 0; c < channels; c++)
                zp[baseOff + c] = zp[baseOff + c] * scale[c] + shift[c];
        }
    }

    /// <summary>Unpatchifies the packed token latent <c>[1, nImg, patch²·aeC]</c> → <c>[1, aeC, gridH·patch, gridW·patch]</c> (upstream <c>view(B,gh,gw,p,p,ae).permute(0,5,1,3,2,4)</c>). For packed feature <c>f</c>: <c>ae = f % aeC, p2 = (f/aeC) % patch, p1 = f / (aeC·patch)</c>.</summary>
    /// <summary>Sums a tensor set's true on-device footprint.</summary>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c> — block-quantized
    /// dtypes report <c>SizeInBytes == 0</c> and would total to zero, silently disabling the streamed path.</remarks>
    private static long SumBytes(IEnumerable<Tensor> tensors)
    {
        long total = 0;
        foreach (Tensor t in tensors) total += t.DType.ComputeByteCount(t.ElementCount);
        return total;
    }

    /// <summary>Total streamable (per-block) bytes for one transformer.</summary>
    private static long SumBlockBytes(Ideogram4Transformer transformer)
    {
        long total = 0;
        for (int i = 0; i < transformer.BlockCount; i++) total += transformer.GetBlock(i).EstimatedWeightBytes;
        return total;
    }

    /// <summary>Builds a block-streaming controller for one transformer, primes it, and hooks its forward pass.</summary>
    /// <remarks>Prefetch depth is halved relative to the single-DiT pipelines: two controllers are live at once and
    /// each briefly holds (prefetch + 2) blocks, so the combined peak is what has to fit.</remarks>
    private BlockStreamingController AttachStreamer(Ideogram4Transformer transformer, long activationReserve)
    {
        IStreamingBlock[] blocks = new IStreamingBlock[transformer.BlockCount];
        for (int i = 0; i < blocks.Length; i++) blocks[i] = transformer.GetBlock(i);

        long avail = Backend.StreamingCache!.QueryAvailableWeightCacheBytes(activationReserve);
        long perBlock = blocks.Length > 0 ? blocks[0].EstimatedWeightBytes : 0;
        // Budget is split across the two live windows, hence the /2; -2 covers the transient overlap where block
        // N+1 becomes resident before N-1 is evicted. Cap at 1 rather than Flux's 2 for the same reason.
        int prefetchAhead = perBlock > 0 ? Math.Clamp((int)(avail / 2 / perBlock) - 2, 0, 1) : 0;

        BlockStreamingController streamer = new BlockStreamingController(
            Backend.StreamingCache, blocks, prefetchAhead: prefetchAhead, retainBehind: 0, backend: Backend);
        transformer.BeforeBlockForward = streamer.BeforeBlockForward;
        streamer.Prime();
        return streamer;
    }

    /// <summary>Per-forward activation/workspace estimate for one Ideogram 4 pass over the unified text+image sequence.</summary>
    /// <remarks>Same shape as <c>FluxPipeline.EstimateFluxActivationReserveBytes</c>: sum the live per-block scratch
    /// (SwiGLU pair, fused QKV, multi-head views, sandwich norms) plus a flat allowance for cuBLAS workspace, fp8→F16
    /// weight-cast buffers and the MRoPE tables. Deliberately generous — an over-estimate costs a slightly smaller
    /// prefetch window, an under-estimate costs an OOM.</remarks>
    private static long EstimateActivationReserveBytes(int seqLen, int embDim, int intermediateSize)
    {
        long swiGluBytes = 2L * seqLen * intermediateSize * 2;   // F16 w1/w3 activations
        long swiGluOutBytes = (long)seqLen * intermediateSize * 2;
        long qkvBytes = 3L * seqLen * embDim * 4;                // F32 fused QKV
        long multiHeadBytes = 3L * seqLen * embDim * 4;          // F32 per-head views
        long normedBytes = 2L * seqLen * embDim * 4;             // F32 sandwich norms
        long residualBytes = (long)seqLen * embDim * 4;          // F32 block input
        long scratch = swiGluBytes + swiGluOutBytes + qkvBytes + multiHeadBytes + normedBytes + residualBytes;
        return scratch + 1024L * 1024 * 1024;
    }

    private static Tensor Unpatchify(Tensor z, int gridH, int gridW, int patch)
    {
        int packedC = (int)z.Shape[2];
        int aeC = packedC / (patch * patch);
        int outH = gridH * patch;
        int outW = gridW * patch;
        Tensor outT = new Tensor(new TensorShape(1, aeC, outH, outW), DType.F32);
        float* src = (float*)z.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int gh = 0; gh < gridH; gh++)
        {
            for (int gw = 0; gw < gridW; gw++)
            {
                long token = (long)gh * gridW + gw;
                long srcBase = token * packedC;
                for (int p1 = 0; p1 < patch; p1++)
                {
                    for (int p2 = 0; p2 < patch; p2++)
                    {
                        int dstH = gh * patch + p1;
                        int dstW = gw * patch + p2;
                        for (int ae = 0; ae < aeC; ae++)
                        {
                            int f = p1 * (aeC * patch) + p2 * aeC + ae;
                            long dstOff = ((long)ae * outH + dstH) * outW + dstW;
                            dst[dstOff] = src[srcBase + f];
                        }
                    }
                }
            }
        }
        return outT;
    }
}
