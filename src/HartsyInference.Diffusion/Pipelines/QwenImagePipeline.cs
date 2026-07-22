using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Qwen-Image text-to-image pipeline. Encodes the prompt through Qwen2.5-VL via <see cref="LlamaStyleEncoder"/>, packs the noisy latent into 2×2 patch tokens, runs <see cref="QwenImageTransformer"/> with flow-match Euler scheduling (dynamic shift), unpacks the predicted velocity back to <c>[B, 16, H, W]</c>, and decodes through the 16-channel Qwen-Image VAE. CFG is applied as a dual-pass when <c>request.CfgScale &gt; 1</c>; otherwise a single conditional forward is used. Transformer + text encoder weights are evicted from VRAM before VAE decode (Phase 3 deviations #18 / #33) so a 30 GB Qwen-Image FP8 stack still leaves room for tiled decode workspace on a 40 GB card.
/// <para>Qwen-Image-Edit: pass a reference image via the <c>editRefImage</c> parameter of <see cref="GenerateFromTokens"/> —
/// its VAE latent is appended as extra frame-1 tokens on every transformer forward (ref-latent conditioning half of the
/// edit model; the Qwen2.5-VL vision-token half is pending a Qwen2.5-VL vision encoder — see the parameter docs).</para></summary>
public sealed unsafe class QwenImagePipeline : DiffusionPipelineBase
{
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly QwenImageVaeDecoder _vaeDecoder;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly Qwen25VlMultimodalEncoder? _multimodalEncoder;
    private readonly QwenImageConfig _config;

    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop
    /// FreeWeights + next-gen re-upload). The Qwen2.5-VL TE cannot coexist with the resident DiT on 24 GB, so a
    /// prompt-cache MISS under this flag frees the DiT first, encodes, then re-preloads — repeat prompts skip both.
    /// Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables) — the miss-path eviction above is what keeps smaller cards viable even with residency on.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used), keyed on (token ids, drop index) — the
    // Krea2Pipeline pattern. A hit skips the whole TE phase (preload + encode + free).
    private int[]? _cachedCondKey;
    private int _cachedCondDrop = -1;
    private Tensor? _cachedCond;
    private int[]? _cachedUncondKey;
    private int _cachedUncondDrop = -1;
    private Tensor? _cachedUncond;
    // Edit-with-vision: the token ids alone don't identify the conditioning (the reference PIXELS feed the
    // vision tower) — a content signature of the ref image joins the cache key. 0 = no vision conditioning.
    private long _cachedEditSig;

    // Edit-reference latent cache (single slot, keyed on the VAE-ref pixel signatures): repeat edit
    // generations (same references, new seed/prompt) reuse the packed ref tokens instead of re-running the
    // VAE encoder. Beyond speed, this is what keeps KEEP_MODELS residency safe for edit: the reference VAE
    // encode needs the DiT evicted first (its Conv2D workspace cannot coexist with a ~20 GB resident DiT on
    // 24 GB cards), so a hit avoids the evict + full re-upload entirely. The tensor stays pinned in the
    // device weight cache until replaced or the pipeline is disposed.
    private Tensor? _cachedPackedRef;
    private long _cachedRefSig;
    private (int H, int W)[] _cachedRefGrids = [];

    /// <summary>Creates a Qwen-Image pipeline. Caller is responsible for the lifetime of the components — they may be reused across pipelines. Img2img is unavailable; use the overload accepting a <see cref="QwenImageVaeEncoder"/> to enable it.</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, QwenImageVaeDecoder vaeDecoder, QwenImageConfig config)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a Qwen-Image pipeline with both VAE halves loaded — required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>).</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder, QwenImageConfig config)
        : this(backend, textEncoder, transformer, vaeDecoder, vaeEncoder, multimodalEncoder: null, config)
    {
    }

    /// <summary>Creates a Qwen-Image-Edit pipeline: adds the Qwen2.5-VL vision-conditioning path. When
    /// <paramref name="multimodalEncoder"/> is non-null and an <c>editRefImage</c> is passed, the prompt token
    /// ids may contain <c>&lt;|image_pad|&gt;</c> placeholders and the TE phase runs the multimodal encode
    /// (vision tower + sectioned M-RoPE) instead of the text-only encode.</summary>
    public QwenImagePipeline(IBackend backend, LlamaStyleEncoder textEncoder,
        QwenImageTransformer transformer, QwenImageVaeDecoder vaeDecoder, QwenImageVaeEncoder? vaeEncoder,
        Qwen25VlMultimodalEncoder? multimodalEncoder, QwenImageConfig config)
        : base(backend)
    {
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _multimodalEncoder = multimodalEncoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized prompt input. The negative prompt tokens are ignored when <see cref="TextToImageRequest.CfgScale"/> ≤ 1. An <see cref="ImageToImageRequest"/> selects img2img: the source is encoded via the Qwen-Image 3D-causal VAE encoder, packed (2×2 patchify), and combined with fresh packed noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c> — requires a <see cref="QwenImageVaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step packed blend + final pixel recomposite, same as Flux). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="promptTokenIds">Conditional prompt token IDs (Qwen2.5-VL vocab).</param>
    /// <param name="negativeTokenIds">Negative-prompt token IDs (same length as <paramref name="promptTokenIds"/> recommended). Used only when <c>CfgScale &gt; 1</c>.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional per-step progress callback.</param>
    /// <param name="promptDropIndex">Number of leading hidden-state positions to drop from the conditional
    /// stream after encoding. Qwen-Image wraps the prompt in a system+user-header template and discards
    /// those prefix tokens' hidden states (diffusers' <c>prompt_template_encode_start_idx</c>). 0 = keep all
    /// (raw-prompt path).</param>
    /// <param name="negativeDropIndex">Same as <paramref name="promptDropIndex"/> for the negative stream.</param>
    /// <param name="editRefImage">Qwen-Image-Edit reference image <c>[1, 3, refH, refW]</c>, F32 in [-1, 1], dims
    /// divisible by 16 (VAE 8× + 2×2 packing; resize upstream — ComfyUI's <c>TextEncodeQwenImageEdit</c> rescales to
    /// ~1 MP area). The reference is VAE-encoded once and its packed latent tokens are appended after the noise
    /// tokens on EVERY transformer forward (both CFG passes, matching diffusers' QwenImageEditPipeline and the
    /// standard ComfyUI edit workflow where positive AND negative conditioning carry the reference) with
    /// frame-axis-1 RoPE — the "reference_latents" conditioning of ComfyUI's <c>comfy_extras/nodes_qwen.py</c>
    /// (TextEncodeQwenImageEdit) + <c>comfy/ldm/qwen_image/model.py</c> (default "index" ref method), analogous to
    /// Flux Kontext. Requires a <see cref="QwenImageVaeEncoder"/>. Caller retains ownership.
    /// <para><b>Scope:</b> this implements the ref-LATENT half of Qwen-Image-Edit only. The other half — feeding the
    /// reference image to the Qwen2.5-VL text encoder as vision tokens inside the edit chat template
    /// (<c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c>) — needs a Qwen2.5-VL VISION encoder,
    /// which the codebase does not have yet (the Qwen3-VL vision tower used by Boogu is a different architecture).
    /// Until then, pass text-only edit-template token ids in <paramref name="promptTokenIds"/>; edit fidelity will be
    /// below the reference implementation because the VL branch cannot "see" the image.</para></param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativeTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null,
        int promptDropIndex = 0,
        int negativeDropIndex = 0,
        IReadOnlyList<Tensor>? editRefImages = null,
        bool editRefTimestepZero = false,
        IReadOnlyList<Tensor>? editRefVisionImages = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a QwenImageVaeEncoder. Construct the pipeline with the overload that accepts one.");
        bool hasEditRefs = editRefImages is { Count: > 0 };
        if (hasEditRefs)
        {
            if (_vaeEncoder is null)
                throw new InvalidOperationException("editRefImages (Qwen-Image-Edit) requires a QwenImageVaeEncoder. Construct the pipeline with the overload that accepts one.");
            foreach (Tensor editRef in editRefImages!)
            {
                if (editRef.Shape.Rank != 4 || editRef.Shape[0] != 1 || editRef.Shape[1] != 3
                    || editRef.Shape[2] % 16 != 0 || editRef.Shape[3] % 16 != 0 || editRef.Shape[2] == 0)
                {
                    throw new ArgumentException(
                        $"each edit reference must be [1, 3, refH, refW] with dims divisible by 16; got {editRef.Shape}.",
                        nameof(editRefImages));
                }
            }
        }

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.QwenImage.Width;
        int height = request.Height ?? GenerationDefaults.QwenImage.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int hPacked = latentH / _config.PatchSize;
        int wPacked = latentW / _config.PatchSize;
        int imgSeqLen = hPacked * wPacked;
        int patchDim = _config.PatchSize * _config.PatchSize * _config.InChannels;
        int steps = request.Steps ?? GenerationDefaults.QwenImage.Steps;
        float cfgScale = request.CfgScale ?? GenerationDefaults.QwenImage.CfgScale;
        bool useCfg = cfgScale > 1.0f;

        // Default-off perf knobs (reference wiring for the fleet — see docs/Checklists/INFERENCE_ACCEL_GRIND.md).
        // HARTSY_CFG_INTERVAL=lo,hi skips the uncond forward outside the normalized-t band (limited-interval
        // guidance); HARTSY_STEP_CACHE=<threshold|1> reuses the block-stack residual across steps when the
        // first block's output has barely drifted (First-Block cache). Unset ⇒ byte-identical baseline path.
        GuidanceInterval cfgInterval = GuidanceInterval.FromEnvironment();
        float stepCacheThreshold = StepCacheEnv.ReadThreshold();
        DeviceFeatureCache? condCache = null;
        DeviceFeatureCache? uncondCache = null;
        if (stepCacheThreshold > 0f)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                int stepCacheCap = StepCacheEnv.ReadCap();
                condCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                if (useCfg) uncondCache = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, StepCacheEnv.ReadPoly(), StepCacheEnv.ReadCalibFile());
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}");
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }
        if (!cfgInterval.IsAlways)
            Logs.Info($"CFG interval ON: guidance applies at normalized t in [{cfgInterval.Start}, {cfgInterval.End}]");
        int cfgSkippedSteps = 0;

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
        if (hasEditRefs) opMode += $" + edit-ref x{editRefImages!.Count}";
        Logs.Info($"Qwen-Image {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // Prompt-embedding cache: identical (tokens, dropIndex) reuse the previous gen's hidden states — the
        // whole TE phase (preload + encode + free) vanishes for repeat prompts (seed-only changes). Reusing the
        // SAME tensor references also keeps any downstream ref-keyed caches warm. Cache holds one cond + one
        // uncond (the last used); a new prompt evicts + disposes.
        long editSig = 0L;
        if (_multimodalEncoder is not null && hasEditRefs)
        {
            IReadOnlyList<Tensor> sigSource = editRefVisionImages is { Count: > 0 } ? editRefVisionImages : editRefImages!;
            foreach (Tensor sigImg in sigSource)
                editSig = editSig * 1000003L ^ ImageSignature.Compute(sigImg);
        }
        bool condHit = _cachedCond is not null && _cachedCondDrop == promptDropIndex
            && _cachedEditSig == editSig
            && _cachedCondKey is not null && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIds);
        bool uncondHit = !useCfg || (_cachedUncond is not null && _cachedUncondDrop == negativeDropIndex
            && _cachedEditSig == editSig
            && _cachedUncondKey is not null && _cachedUncondKey.AsSpan().SequenceEqual(negativeTokenIds));
        Tensor condHidden;
        Tensor? uncondHidden = null;
        if (condHit && uncondHit)
        {
            condHidden = _cachedCond!;
            if (useCfg) uncondHidden = _cachedUncond;
            Logs.Info("[QwenImage] prompt-embedding cache hit — TE phase skipped");
        }
        else
        {
            // Edit-with-vision: the templated ids contain <|image_pad|> placeholders — run the multimodal
            // encode (vision tower + spliced embeds + sectioned M-RoPE). Both cond AND uncond carry the
            // image (the standard ComfyUI edit workflow feeds the reference to both conditionings).
            // The edit-plus recipe conditions the vision tower on ~384²-area rescales of the references
            // while the VAE reference latents use the ~1MP rescale — the caller passes both sets. Vision
            // images are [0,1] (Qwen25VlImageProcessor contract); the fallback converts the [-1,1]
            // VAE-contract references on the host (one-time, off the step loop).
            bool visionEncode = _multimodalEncoder is not null && hasEditRefs
                && Array.IndexOf(promptTokenIds, Qwen25VlMultimodalEncoder.ImageTokenId) >= 0;

            // The TE cannot coexist with the resident DiT (HARTSY_KEEP_MODELS); evict for this
            // new-prompt generation and re-preload below. The vision tower is staged with the language
            // tower on the vision path (and freed with it below) — the double encode (cond + uncond) would
            // otherwise auto-promote its weights into the resident cache, permanently shrinking the DiT budget.
            EvictResidentTransformer("text-encode phase");
            Backend.PreloadWeights(_textEncoder.EnumerateWeights());
            if (visionEncode)
                Backend.PreloadWeights(_multimodalEncoder!.EnumerateVisionWeights());
            List<Tensor> visionFallbacks = new();
            Tensor[] visionImages = [];
            if (visionEncode)
            {
                if (editRefVisionImages is { Count: > 0 })
                {
                    visionImages = [.. editRefVisionImages];
                }
                else
                {
                    foreach (Tensor editRef in editRefImages!)
                    {
                        Tensor conv = new Tensor(editRef.Shape, DType.F32);
                        float* srcPx = (float*)editRef.DataPointer;
                        float* dstPx = (float*)conv.DataPointer;
                        long n = editRef.ElementCount;
                        for (long i = 0; i < n; i++) dstPx[i] = (srcPx[i] + 1.0f) * 0.5f;
                        visionFallbacks.Add(conv);
                    }
                    visionImages = [.. visionFallbacks];
                }
            }

            condHidden = visionEncode
                ? _multimodalEncoder!.Encode(Backend, promptTokenIds, visionImages)
                : _textEncoder.Encode(Backend, [promptTokenIds]);
            if (promptDropIndex > 0)
            {
                Tensor trimmed = DropPrefixHiddenStates(condHidden, promptDropIndex);
                condHidden.Dispose();
                condHidden = trimmed;
            }

            if (useCfg)
            {
                bool negVision = visionEncode
                    && Array.IndexOf(negativeTokenIds, Qwen25VlMultimodalEncoder.ImageTokenId) >= 0;
                uncondHidden = negVision
                    ? _multimodalEncoder!.Encode(Backend, negativeTokenIds, visionImages)
                    : _textEncoder.Encode(Backend, [negativeTokenIds]);
                if (negativeDropIndex > 0)
                {
                    Tensor trimmed = DropPrefixHiddenStates(uncondHidden, negativeDropIndex);
                    uncondHidden.Dispose();
                    uncondHidden = trimmed;
                }
            }

            foreach (Tensor conv in visionFallbacks) conv.Dispose();

            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (txt seqLen={condHidden.Shape[1]})");

            Backend.FreeWeights(_textEncoder.EnumerateWeights());
            if (visionEncode)
                Backend.FreeWeights(_multimodalEncoder!.EnumerateVisionWeights());

            // Materialize the conditioning on the host, then reclaim every encoder intermediate. The Qwen2.5-VL
            // encoder leaves hundreds of device-cached activations that otherwise linger until GC finalization —
            // they'd hold multi-GB into the DiT phase. condHidden/uncondHidden may still be live GPU activations
            // (the drop-index path already copied them on the host, the raw path did not); touching DataPointer
            // forces the D2H sync + cache eviction, making them safe across the FreeActivations calls below.
            _ = condHidden.DataPointer;
            if (uncondHidden is not null) _ = uncondHidden.DataPointer;
            Backend.FreeActivations();

            _cachedCond?.Dispose();
            _cachedCond = condHidden;
            _cachedCondKey = (int[])promptTokenIds.Clone();
            _cachedCondDrop = promptDropIndex;
            _cachedEditSig = editSig;
            if (useCfg)
            {
                _cachedUncond?.Dispose();
                _cachedUncond = uncondHidden;
                _cachedUncondKey = (int[])negativeTokenIds.Clone();
                _cachedUncondDrop = negativeDropIndex;
            }
        }

        TensorShape latentShape = new TensorShape(1, _config.InChannels, latentH, latentW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, patchDim);

        // Qwen-Image scheduler constants (Qwen/Qwen-Image scheduler_config.json) — NOT Flux's defaults:
        // max_image_seq_len 8192 (Flux 4096), max_shift 0.9 (Flux 1.15); base 256 / 0.5 match.
        // shift_terminal 0.02 is applied inside SetTimesteps below.
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(
                imgSeqLen, baseSeqLen: 256, maxSeqLen: 8192, baseShift: 0.5f, maxShift: 0.9f, shiftTerminal: 0.02f);
        scheduler.SetTimesteps(steps);

        // Build initial packed latent — t2i: packed noise * initSigma; img2img: encode + pack + AddNoise
        // at sigma[startStep]. Masked inpaint keeps the packed source + packed mask for per-step blend.
        // A prompt-cache hit skips the TE phase's DiT eviction, so under KEEP_MODELS the DiT can still be
        // resident here — the source VAE encode cannot run beside it (see EvictResidentTransformer).
        if (isImg2Img)
            EvictResidentTransformer("img2img source VAE encode");
        (Tensor packedLatent, Tensor? packedSourceLatent) =
            BuildInitialPackedLatent(request, scheduler, latentShape, packedShape, latentH, latentW, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? packedMask = null;
        if (isMaskedInpaint)
        {
            if (_config.PatchSize != 2)
                throw new InvalidOperationException(
                    $"Masked inpaint requires patch size 2 (PackLatentMask2x2); config has {_config.PatchSize}.");
            Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
            packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latentH, latentW);
            latentMask.Dispose();
        }

        // Qwen-Image-Edit reference: VAE-encode + pack ONCE to [1, refSeq, 64]. Like Flux Kontext, the ref
        // tokens are NOT noised or stepped — they re-enter the transformer identically at every step and are
        // dropped from the velocity output (comfy/ldm/qwen_image/model.py: process_img(ref, index=1) + concat).
        Tensor? packedEditRef = null;                    // all references, row-concatenated in Picture order
        (int H, int W)[] refGrids = [];
        if (hasEditRefs)
        {
            long refSig = 0L;
            foreach (Tensor editRef in editRefImages!)
                refSig = refSig * 1000003L ^ ImageSignature.Compute(editRef);
            if (_cachedPackedRef is not null && _cachedRefSig == refSig)
            {
                packedEditRef = _cachedPackedRef;
                refGrids = _cachedRefGrids;
                Logs.Info($"[QwenImage] edit-reference latent cache hit — VAE encode skipped ({packedEditRef.Shape[1]} ref tokens)");
            }
            else
            {
                // Same-prompt edit re-runs reach here with the DiT still resident (the prompt cache skipped
                // the TE phase and its eviction); the reference VAE encode cannot run beside it.
                EvictResidentTransformer("edit-reference VAE encode");
                // Stage the encoder weights for the encode only: multi-ref runs the encoder once per image,
                // and the second pass would auto-promote its weights into the resident cache for good.
                Backend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
                Stopwatch refSw = Stopwatch.StartNew();
                List<Tensor> packedRefs = new(editRefImages.Count);
                refGrids = new (int, int)[editRefImages.Count];
                int totalRefTokens = 0;
                for (int r = 0; r < editRefImages.Count; r++)
                {
                    Tensor editRef = editRefImages[r];
                    int refLatentH = (int)editRef.Shape[2] / 8;
                    int refLatentW = (int)editRef.Shape[3] / 8;
                    refGrids[r] = (refLatentH / _config.PatchSize, refLatentW / _config.PatchSize);
                    Tensor refUnpacked = _vaeEncoder!.Encode(Backend, editRef);
                    Tensor packedRef = PackLatent(refUnpacked, refLatentH, refLatentW, _config.InChannels, _config.PatchSize);
                    refUnpacked.Dispose();
                    packedRefs.Add(packedRef);
                    totalRefTokens += (int)packedRef.Shape[1];
                }
                if (packedRefs.Count == 1)
                {
                    packedEditRef = packedRefs[0];
                }
                else
                {
                    packedEditRef = new Tensor(new TensorShape(1, totalRefTokens, patchDim), DType.F32);
                    float* dstRef = (float*)packedEditRef.DataPointer;
                    long rowOff = 0;
                    foreach (Tensor packedRef in packedRefs)
                    {
                        long bytes = packedRef.ElementCount * sizeof(float);
                        Buffer.MemoryCopy((void*)packedRef.DataPointer, dstRef + rowOff * patchDim, bytes, bytes);
                        rowOff += packedRef.Shape[1];
                        packedRef.Dispose();
                    }
                }
                refSw.Stop();
                Backend.FreeWeights(_vaeEncoder.EnumerateWeights());
                Logs.Info($"Qwen-Image-Edit references encoded in {refSw.ElapsedMilliseconds}ms: " +
                    $"{editRefImages.Count} image(s), {totalRefTokens} ref tokens.");

                // Replace the single-slot cache. Evict the old pin BEFORE disposing: the weight cache is
                // keyed on the tensor object, and disposing a still-registered pin would leak its device
                // allocation into every later gen.
                if (_cachedPackedRef is not null)
                {
                    Backend.FreeWeights(new[] { _cachedPackedRef });
                    _cachedPackedRef.Dispose();
                }
                _cachedPackedRef = packedEditRef;
                _cachedRefSig = refSig;
                _cachedRefGrids = refGrids;
            }
        }

        // Materialize every tensor that must survive across steps on the host, then reclaim the VAE-encode /
        // packing intermediates BEFORE the DiT re-upload below — the fp8 DiT does not fit beside a 1MP encode's
        // conv workspace on 24 GB cards. The per-step FreeActivations below frees device buffers WITHOUT a D2H
        // sync-back, so anything still device-only here would be silently lost — the t2i initSigma path leaves
        // packedLatent as a live Backend.Scale output on the GPU.
        _ = packedLatent.DataPointer;
        if (packedSourceLatent is not null) _ = packedSourceLatent.DataPointer;
        if (packedMask is not null) _ = packedMask.DataPointer;
        if (packedEditRef is not null) _ = packedEditRef.DataPointer;
        Backend.FreeActivations();

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        _ditResident = true;

        // Pin the step-invariant ref tokens as a device-resident weight: they re-enter the per-step device
        // concat below, and a pageable host re-upload every step is an internally-syncing copy. No-op on a
        // ref-latent cache hit (still pinned from the previous generation).
        if (packedEditRef is not null)
            Backend.PreloadWeights(new List<Tensor> { packedEditRef });

        Logs.Info("Starting Qwen-Image denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Drain-free fast path (t2i / img2img / EDIT): the latent stays device-resident across the whole
        // loop, CFG combine + Euler run as ONE in-place device op (CfgEulerStep: z += (neg + gw·(pos−neg))·dt —
        // for gw=cfgScale this IS uncond + cfg·(cond−uncond), and gw=1 degenerates to the plain Euler step),
        // and FreeActivations never runs mid-loop. Edit joins the fast path via a per-step DEVICE row-concat
        // of [noise ; pinned ref tokens] (the old host ConcatPackedSeqDim forced a full D2H/H2D round trip +
        // host Euler + a per-step sweep — the dominant edit overhead). Masked inpaint keeps the host branch
        // (per-step host blend must read the latent anyway).
        bool drainFree = !isMaskedInpaint;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];
            float normalizedT = t / 1000.0f;

            // Edit mode: append the static ref tokens to the current noisy tokens for THIS forward only —
            // the transformer slices them back off the velocity, so the scheduler below never sees them.
            // Drain-free: device row-concat (two async DtoD copies); host branch keeps the host concat.
            Tensor transformerInput;
            if (packedEditRef is null)
            {
                transformerInput = packedLatent;
            }
            else if (drainFree)
            {
                transformerInput = new Tensor(
                    new TensorShape(1, packedLatent.Shape[1] + packedEditRef.Shape[1], patchDim), DType.F32);
                Backend.Concat(transformerInput, new Tensor[] { packedLatent, packedEditRef }, 1);
            }
            else
            {
                transformerInput = ConcatPackedSeqDim(packedLatent, packedEditRef);
            }

            // Limited-interval guidance: outside the band the uncond forward is pure waste (the paper measures
            // a quality IMPROVEMENT from dropping it there), so the step degenerates to the cond-only path.
            bool cfgThisStep = useCfg && cfgInterval.Applies(normalizedT);
            if (useCfg && !cfgThisStep) cfgSkippedSteps++;

            if (drainFree)
            {
                Tensor condPred = _transformer.Forward(Backend, transformerInput, condHidden, normalizedT, hPacked, wPacked, refGrids, editRefTimestepZero, condCache);
                if (cfgThisStep)
                {
                    Tensor uncondPred = _transformer.Forward(Backend, transformerInput, uncondHidden!, normalizedT, hPacked, wPacked, refGrids, editRefTimestepZero, uncondCache);
                    Backend.CfgEulerStep(packedLatent, condPred, uncondPred, cfgScale, scheduler.Dt(i));
                    uncondPred.Dispose();
                }
                else
                {
                    Backend.CfgEulerStep(packedLatent, condPred, condPred, 1.0f, scheduler.Dt(i));
                }
                condPred.Dispose();
                if (transformerInput != packedLatent) transformerInput.Dispose();
            }
            else
            {
                Tensor noisePred;
                if (cfgThisStep)
                {
                    Tensor condPred = _transformer.Forward(Backend, transformerInput, condHidden, normalizedT, hPacked, wPacked, refGrids, editRefTimestepZero, condCache);
                    Tensor uncondPred = _transformer.Forward(Backend, transformerInput, uncondHidden!, normalizedT, hPacked, wPacked, refGrids, editRefTimestepZero, uncondCache);
                    noisePred = CfgHelper.ApplyCfg(uncondPred, condPred, cfgScale);
                    uncondPred.Dispose();
                    condPred.Dispose();
                }
                else
                {
                    noisePred = _transformer.Forward(Backend, transformerInput, condHidden, normalizedT, hPacked, wPacked, refGrids, editRefTimestepZero, condCache);
                }
                if (transformerInput != packedLatent) transformerInput.Dispose();

                Tensor newLatent = new Tensor(packedShape, DType.F32);
                scheduler.Step(newLatent, noisePred, packedLatent, i);
                noisePred.Dispose();
                packedLatent.Dispose();
                packedLatent = newLatent;
            }

            // Masked-inpaint blend in packed form: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source latent at the next step's
            // sigma. Final step blends with the clean source — no further denoising follows.
            if (packedMask is not null && packedSourceLatent is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshUnpackedNoise = SeedGenerator.CreateNoise(latentShape, seed + nextStep);
                    Tensor freshPackedNoise = PackLatent(freshUnpackedNoise, latentH, latentW, _config.InChannels, _config.PatchSize);
                    freshUnpackedNoise.Dispose();
                    noisedSource = new Tensor(packedShape, DType.F32);
                    scheduler.AddNoise(noisedSource, packedSourceLatent, freshPackedNoise, nextStep);
                    freshPackedNoise.Dispose();
                }
                else
                {
                    noisedSource = packedSourceLatent;
                }
                MaskBlendUtilities.BlendPackedInPlace(packedLatent, noisedSource, packedMask);
                if (noisedSource != packedSourceLatent) noisedSource.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));

            // Reclaim GPU-resident activation buffers between steps ONLY on the host paths (their
            // scheduler.Step/blends leave the latent host-resident, so nothing device-only is lost). The
            // drain-free path skips this: every intermediate is explicitly disposed, and the latent — kept
            // device-resident by the in-place CfgEulerStep — would be silently reverted to a stale host copy.
            if (!drainFree)
            {
                Backend.FreeActivations();
            }
        }

        // condHidden/uncondHidden and packedEditRef (the pinned ref-latent cache slot) are cross-generation
        // caches — not disposed here; DisposeCore / cache replacement release them.
        packedSourceLatent?.Dispose();
        packedMask?.Dispose();

        // Perf-knob accounting for benchmark records: forwards actually run vs the uncached/ungated baseline.
        if (condCache is not null)
        {
            string uncondStats = uncondCache is not null
                ? $"; uncond {uncondCache.Computes} computes / {uncondCache.Reuses} reuses"
                : string.Empty;
            Logs.Info($"Step cache: cond {condCache.Computes} computes / {condCache.Reuses} reuses{uncondStats}");
            condCache.Dispose();
            uncondCache?.Dispose();
        }
        if (cfgSkippedSteps > 0)
            Logs.Info($"CFG interval: uncond forward skipped on {cfgSkippedSteps}/{steps - startStep} steps");

        QwenImageTransformer.DumpFinalLatent(packedLatent);

        Backend.Sync();
        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }

        Tensor unpacked = UnpackLatent(packedLatent, latentH, latentW, _config.InChannels, _config.PatchSize);
        packedLatent.Dispose();

        // Reclaim the denoise-loop activation pool BEFORE the decoder preload: the drain-free loop never
        // frees mid-loop, and with the DiT left resident (KEEP_MODELS) those buffers are the difference
        // between the 1024² decode's conv workspace fitting or OOMing on 24 GB cards. Safe here: UnpackLatent
        // above already forced the final latent's D2H sync, and everything else that survives is host-side
        // or pinned in the weight cache. trimPool so the free-VRAM gate below measures real availability.
        Backend.FreeActivations(trimPool: true);

        // Decode VRAM gate: the full-res 3D-causal decode holds every intermediate feature map until the
        // final FreeActivations (~4 KB/pixel live at peak, plus the banded-im2col Conv2D workspace). When
        // the resident DiT leaves less than that, evict it — the next generation's re-upload is strictly
        // cheaper than OOM-ing the decode (a 19 GB fp8 DiT + 1024² decode do NOT fit together on 24 GB).
        // Cards with real headroom keep residency; non-CUDA backends have no streaming cache and skip this.
        if (_ditResident && Backend.StreamingCache is not null)
        {
            long decodeReserveBytes = (long)width * height * 5120L;
            if (Backend.StreamingCache.QueryAvailableWeightCacheBytes(decodeReserveBytes) == 0)
                EvictResidentTransformer($"VAE decode ({width}x{height} needs ~{decodeReserveBytes / (1024 * 1024)} MB free)");
        }

        // The QwenImage VAE is the WAN-2.1-family 3D causal autoencoder (decoder.conv1/upsamples/…), NOT a
        // diffusers AutoencoderKL — it must decode through QwenImageVaeDecoder (same as AnimaPipeline). The
        // per-channel latent un-normalization (latents_mean/std from VaeConfig.QwenImage) happens INSIDE
        // QwenImageVaeDecoder.Decode → UndoScaling, so we pass the unpacked latent directly (no scalar
        // shift/scale here — VaeConfig.QwenImage's scale=1/shift=0 made the old ApplyVaeShiftScale a no-op
        // anyway, and it skipped the real per-channel step).
        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        Logs.Info("Decoding latents to image (QwenImage 3D-causal VAE)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(Backend, unpacked);
        unpacked.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        // Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the budget of whatever generation runs next.
        Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Qwen-Image {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Evicts the KEEP_MODELS-resident DiT so a VRAM-heavy phase (TE encode, reference/img2img VAE encode) has headroom; the denoise-loop PreloadWeights restores residency (PreloadWeights/FreeWeights symmetry). No-op when not resident.</summary>
    private void EvictResidentTransformer(string reason)
    {
        if (!_ditResident) return;
        Logs.Debug($"[QwenImage] evicting resident DiT for {reason}");
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        _ditResident = false;
    }

    /// <summary>Releases the cross-generation caches: the prompt-embedding tensors and the pinned edit-reference latent (device pin evicted before dispose — the weight cache is keyed on the tensor object).</summary>
    protected override void DisposeCore()
    {
        if (_cachedPackedRef is not null)
        {
            Backend.FreeWeights(new[] { _cachedPackedRef });
            _cachedPackedRef.Dispose();
            _cachedPackedRef = null;
        }
        _cachedCond?.Dispose();
        _cachedCond = null;
        _cachedCondKey = null;
        _cachedUncond?.Dispose();
        _cachedUncond = null;
        _cachedUncondKey = null;
    }

    /// <summary>Builds the initial packed latent for Qwen-Image denoising. T2I: fresh Gaussian noise packed and scaled by the scheduler's initial sigma. Img2img: source encoded via the Qwen-Image 3D-causal VAE encoder (already per-channel normalized to the transformer's latent space), packed via <see cref="PackLatent"/>, and combined with fresh packed noise via flow-matching <c>AddNoise</c>: <c>noisy = (1-sigma) * source + sigma * noise</c>.
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the packed source latent is returned as the second tuple element for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor packedLatent, Tensor? packedSourceLatent) BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape latentShape,
        TensorShape packedShape,
        int latentH, int latentW, int seed, int startStep,
        bool keepSourceLatent)
    {
        Tensor unpackedNoise = SeedGenerator.CreateNoise(latentShape, seed);
        Tensor packedNoise = PackLatent(unpackedNoise, latentH, latentW, _config.InChannels, _config.PatchSize);
        unpackedNoise.Dispose();

        if (request is ImageToImageRequest img2img)
        {
            // Stage the encoder weights for this one encode — an unstaged pass auto-promotes them into the
            // resident weight cache on repeat gens and permanently shrinks the DiT budget.
            Backend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceUnpacked = _vaeEncoder.Encode(Backend, img2img.SourceImage);  // [1, 16, latentH, latentW]
            vaeEncSw.Stop();
            Backend.FreeWeights(_vaeEncoder.EnumerateWeights());
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor sourcePacked = PackLatent(sourceUnpacked, latentH, latentW, _config.InChannels, _config.PatchSize);
            sourceUnpacked.Dispose();

            Tensor result = new Tensor(packedShape, DType.F32);
            scheduler.AddNoise(result, sourcePacked, packedNoise, startStep);
            packedNoise.Dispose();
            if (keepSourceLatent)
            {
                return (result, sourcePacked);
            }
            sourcePacked.Dispose();
            return (result, null);
        }

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(packedShape, DType.F32);
            Backend.Scale(scaled, packedNoise, initSigma);
            packedNoise.Dispose();
            return (scaled, null);
        }
        return (packedNoise, null);
    }

    /// <summary>Concatenates two packed-form tensors <c>[1, Sa, D]</c> and <c>[1, Sb, D]</c> along the SEQUENCE dim
    /// → <c>[1, Sa+Sb, D]</c>. Used by Qwen-Image-Edit to append the packed reference-latent tokens after the packed
    /// noise tokens before the transformer (same as Flux Kontext). Both inputs must share batch and feature dim; F32 only.</summary>
    private static Tensor ConcatPackedSeqDim(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 3 || b.Shape.Rank != 3 || a.Shape[0] != b.Shape[0] || a.Shape[2] != b.Shape[2])
            throw new ArgumentException($"ConcatPackedSeqDim requires [B, S, D] tensors with matching B and D; got {a.Shape} and {b.Shape}.");
        long batch = a.Shape[0];
        long seqA = a.Shape[1];
        long seqB = b.Shape[1];
        long dim = a.Shape[2];
        Tensor output = new Tensor(new TensorShape(batch, seqA + seqB, dim), DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* op = (float*)output.DataPointer;
        for (long bi = 0; bi < batch; bi++)
        {
            long aBytes = seqA * dim * sizeof(float);
            long bBytes = seqB * dim * sizeof(float);
            Buffer.MemoryCopy(ap + bi * seqA * dim, op + bi * (seqA + seqB) * dim, aBytes, aBytes);
            Buffer.MemoryCopy(bp + bi * seqB * dim, op + bi * (seqA + seqB) * dim + seqA * dim, bBytes, bBytes);
        }
        return output;
    }

    /// <summary>Drops the first <paramref name="drop"/> sequence positions from a <c>[1, seq, hidden]</c>
    /// F32 hidden-state tensor, returning <c>[1, seq-drop, hidden]</c>. Used to discard the system+user-header
    /// template prefix from Qwen-Image text conditioning (the kept tail = prompt content + assistant suffix,
    /// matching diffusers' <c>split_hidden_states[drop_idx:]</c>). No-op clone guard if drop is out of range.</summary>
    private static Tensor DropPrefixHiddenStates(Tensor hidden, int drop)
    {
        long batch = hidden.Shape[0];
        long seq = hidden.Shape[1];
        long hiddenDim = hidden.Shape[2];
        long elem = hidden.DType.SizeInBytes;
        long rowBytes = hiddenDim * elem;
        // Guard: if the drop would empty the sequence (or batch isn't 1), copy the whole tensor unchanged
        // rather than drop — the caller always disposes the input and adopts our return value.
        long effectiveDrop = (drop <= 0 || drop >= seq || batch != 1) ? 0 : drop;
        long newSeq = seq - effectiveDrop;
        Tensor result = new Tensor(new TensorShape(batch, newSeq, hiddenDim), hidden.DType);
        byte* src = (byte*)hidden.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy(src + effectiveDrop * rowBytes, dst, newSeq * rowBytes, newSeq * rowBytes);
        return result;
    }

    /// <summary>Packs a latent tensor from <c>[B, C, H, W]</c> to <c>[B, (H/p)*(W/p), p² * C]</c> by interleaving p×p spatial blocks into the feature dim. Diffusers uses <c>view → permute(0,2,4,1,3,5) → reshape</c>; we replicate the resulting layout where each p×p patch contributes channels C × p² to the token, ordered <c>(c, py, px)</c> so the inverse <see cref="UnpackLatent"/> is exact.</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w, int channels, int patchSize)
    {
        int batch = (int)latent.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape packedShape = new TensorShape(batch, seqLen, patchDim);
        Tensor packed = new Tensor(packedShape, DType.F32);

        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long outBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long inChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = outBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[patchBase + py * patchSize + px] = inPtr[inChannelBase + (long)row * w + col];
                            }
                        }
                    }
                }
            }
        }

        return packed;
    }

    /// <summary>Inverse of <see cref="PackLatent"/>.</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w, int channels, int patchSize)
    {
        int batch = (int)packed.Shape[0];
        int hPacked = h / patchSize;
        int wPacked = w / patchSize;
        int patchDim = channels * patchSize * patchSize;
        int seqLen = hPacked * wPacked;

        TensorShape unpackedShape = new TensorShape(batch, channels, h, w);
        Tensor unpacked = new Tensor(unpackedShape, DType.F32);

        float* inPtr = (float*)packed.DataPointer;
        float* outPtr = (float*)unpacked.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < hPacked; ph++)
            {
                for (int pw = 0; pw < wPacked; pw++)
                {
                    int seqIdx = ph * wPacked + pw;
                    long inBase = ((long)b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        long outChannelBase = ((long)b * channels + c) * h * w;
                        long patchBase = inBase + (long)c * patchSize * patchSize;

                        for (int py = 0; py < patchSize; py++)
                        {
                            for (int px = 0; px < patchSize; px++)
                            {
                                int row = ph * patchSize + py;
                                int col = pw * patchSize + px;
                                outPtr[outChannelBase + (long)row * w + col] = inPtr[patchBase + py * patchSize + px];
                            }
                        }
                    }
                }
            }
        }

        return unpacked;
    }

    /// <summary>Applies the diffusers-style VAE input transform: <c>(latent / scaling_factor) + shift_factor</c>. The decoder expects this de-normalized input; the scaling and shift constants come from the model card.</summary>
    private static void ApplyVaeShiftScale(Tensor output, Tensor input, float shift, float scale)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long count = output.ElementCount;
        float invScale = 1.0f / scale;
        for (long i = 0; i < count; i++)
            outPtr[i] = inPtr[i] * invScale + shift;
    }
}
