using HartsyInference.Diffusion.Sampling;
using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Flux text-to-image and image-to-image pipeline. Orchestrates CLIP-L pooled + T5-XXL text encoding → FluxTransformer denoising with flow matching → VAE decode → RGB image output. Supports Dev (guidance embedding) and Schnell (distilled, 1-4 steps) modes.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromTokens"/>. Requires a <see cref="VaeEncoder"/> on construction. The img2img path encodes the source via the 16-channel Flux VAE, packs the latent (2×2 patchify), and injects flow-matching noise at the timestep selected by <c>Strength</c>.</para>
/// </summary>
public sealed unsafe class FluxPipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipL;
    private readonly T5TextEncoder _t5;
    private readonly FluxTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly FluxConfig _config;

    /// <summary>Keeps the DiT weights GPU-resident across generations on the eager (non-streaming) path — skips the post-loop FreeWeights + next-gen re-upload. A prompt-cache MISS frees the DiT before the T5 encode (the TE cannot always coexist with the resident DiT), then the loop re-preloads. Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables); the streaming (low-VRAM) path always evicts.</summary>
    /// <summary>True when the current residency is the sharded asymmetric layout (shared + [0, split) on <see cref="DiffusionPipelineBase.Backend"/>, [split, BlockCount) on <see cref="DiffusionPipelineBase.DitShardBackend"/>) rather than the whole DiT on the primary — the free path must mirror whichever preload shape actually ran or the shard backend's range leaks.</summary>
    private bool _ditShardResident;

    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used), keyed on the CLIP-L and T5 token ids —
    // the Krea2/Chroma pattern. A hit skips the whole CLIP+T5 phase (preload + encode + free). Cached
    // tensors are host-materialized so they survive the per-gen FreeActivations sweeps.
    private int[]? _cachedCondKey;
    private Tensor? _cachedClipPooled;
    private Tensor? _cachedT5;
    private int[]? _cachedNegKey;
    private Tensor? _cachedNegClipPooled;
    private Tensor? _cachedNegT5;

    /// <summary>HARTSY_FLUX_STATS=1 re-enables the per-tensor debug statistics (min/max/mean/NaN scans and per-channel means). Each scan is a full host read of a device-resident tensor — a forced D2H sync that serializes the denoise loop — so they are strictly opt-in diagnostics, never on by default.</summary>
    private static readonly bool StatsEnabled =
        Environment.GetEnvironmentVariable("HARTSY_FLUX_STATS") == "1";

    /// <summary>Creates a new Flux pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public FluxPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, FluxConfig config)
        : this(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder: null, config)
    {
    }

    /// <summary>Creates a new Flux pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public FluxPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, FluxConfig config)
        : base(backend)
    {
        _clipL = clipL;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
    }

    /// <summary>Encodes arbitrary text through this pipeline's own T5-XXL encoder — the same encoder instance and backend the base prompt uses, so a region's caption lands in the identical embedding space. For regional/object prompt conditioning built by the caller (<see cref="Prompting.RegionalPromptResolver"/>'s <c>encodeRegion</c> delegate); the recipe pipeline owns the T5 tokenizer, this owns the T5 encoder, so neither side alone can do this. Returns a <c>[1, L, hidden]</c> tensor; disposal is the caller's responsibility (<see cref="Prompting.RegionalPromptResolver.DisposeRegions"/> covers it once the region is attached to a <see cref="Prompting.RegionalPlan"/>).</summary>
    public Tensor EncodeRegionText(int[] tokenIds, int[]? attentionMask = null) =>
        _t5.Encode(TextEncoderBackend, [tokenIds], attentionMask is null ? null : [attentionMask]);

    /// <summary>Generates an image from pre-tokenized input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    ///   <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial packed latent = noise scaled by initSigma; denoise from step 0).</item>
    ///   <item><see cref="ImageToImageRequest"/> → image-to-image. The source image is encoded via the 16-channel Flux VAE, packed (2×2 patchify), and combined with fresh packed noise via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// Strength=0 short-circuits to byte-identical pass-through.
    /// <para>True-CFG (diffusers <c>true_cfg_scale</c>): when <paramref name="trueCfgScale"/> &gt; 1 and a negative T5 token stream is supplied, a second unconditional transformer forward runs each step against the negative CLIP-pooled + negative T5 conditioning and the two velocity predictions are combined via standard CFG (<c>neg + scale·(pos − neg)</c>). This is layered ON TOP of Flux's embedded distilled guidance, which still rides along on both passes. When the trigger is not met, the path is byte-identical to the single-pass loop.</para></summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int promptEosPositionL,
        int[] promptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        TextToImageRequest request,
        float guidanceScale = 3.5f,
        Action<GenerationProgress>? onProgress = null,
        Tensor? controlImage = null,
        RegionalPlan? regionalPlan = null,
        int[]? negPromptTokenIdsL = null,
        int negPromptEosPositionL = 0,
        int[]? negPromptTokenIdsT5 = null,
        int[]? negPromptAttentionMaskT5 = null,
        float trueCfgScale = 1.0f,
        Tensor? kontextRefImage = null,
        Tensor? reduxImageEmbeds = null,
        float reduxApplyStartFraction = 0f,
        IReadOnlyList<Adapters.FluxControlNetConditioning>? fluxControlNets = null)
    {
        ThrowIfDisposed();
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        bool hasFluxCn = fluxControlNets is { Count: > 0 };
        if (hasFluxCn)
        {
            if (_vaeEncoder is null)
                throw new InvalidOperationException("Flux ControlNet conditioning requires a VaeEncoder (the control image is VAE-encoded). Construct the pipeline with the overload that accepts one.");
            if (kontextRefImage is not null)
                throw new InvalidOperationException("Flux ControlNet conditioning cannot be combined with a Kontext reference image.");
        }

        // FLUX.1 Tools detection: vanilla Flux has x_embed input dim 64 (16 latent channels
        // × 2×2 packing). Canny / Depth have 128 (64 noise + 64 packed VAE-encoded control,
        // concatenated along the feature dim before the transformer). FLUX.1 Fill has 384
        // (64 noise + 320 = packed masked-image latent + packed mask). Detect Tools as any
        // x_embedder wider than the vanilla 64.
        bool isToolsModel = _transformer.XEmbedInputDim > 64;
        // Fill requires masked-image + mask conditioning (in_channels 384), which is a
        // distinct prep from the single-control-image concat used by Canny/Depth.
        bool isFillModel = _transformer.XEmbedInputDim >= 384;
        if (isFillModel)
        {
            if (request is not ImageToImageRequest fillReq || fillReq.Mask is null)
            {
                throw new InvalidOperationException(
                    "This Flux checkpoint is FLUX.1 Fill (x_embedder input dim 384) and requires a source image + mask. " +
                    "Pass an ImageToImageRequest with SourceImage and Mask (Strength 1.0 for a standard fill/outpaint).");
            }
            if (fillReq.SourceImage.DType != DType.F32)
            {
                throw new ArgumentException($"FLUX.1 Fill SourceImage must be F32 in [-1, 1]; got {fillReq.SourceImage.DType}.", nameof(request));
            }
            if (_vaeEncoder is null)
            {
                throw new InvalidOperationException(
                    "FLUX.1 Fill requires a VaeEncoder to encode the masked source image. Construct the pipeline with the overload that accepts one.");
            }
            if (controlImage is not null)
            {
                throw new InvalidOperationException(
                    "FLUX.1 Fill conditions on SourceImage + Mask; controlImage is only for Canny / Depth checkpoints.");
            }
        }
        else if (isToolsModel)
        {
            if (controlImage is null)
            {
                throw new InvalidOperationException(
                    "This Flux checkpoint is a FLUX.1 Tools variant (Canny / Depth — x_embedder input dim is 128) and requires a control image. Pass one via the controlImage parameter; for Canny it should be the canny-edge map of the user's reference image, for Depth a depth-map estimate.");
            }
            if (_vaeEncoder is null)
            {
                throw new InvalidOperationException(
                    "FLUX.1 Tools variants require a VaeEncoder to encode the control image. Construct the pipeline with the overload that accepts one.");
            }
        }
        else if (controlImage is not null)
        {
            throw new InvalidOperationException(
                "Control image was supplied but this Flux checkpoint is the vanilla variant (x_embedder input dim is 64). Use a FLUX.1 Canny / Depth / Fill checkpoint or remove the control image.");
        }

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, _, int width, int height) = GenerationDefaults.FluxDev.Resolve(request);
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
        // Fill conditions the transformer on the masked image + mask through the wide x_embedder;
        // the per-step latent blend below is the VANILLA-model inpaint emulation and must not also run.
        bool isMaskedInpaint = maskPixel is not null && !isFillModel;

        string baseMode = _config.GuidanceEmbed ? "Dev" : "Schnell";
        string opMode = isFillModel ? $"fill (start={startStep}/{steps})"
                       : isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                       : isImg2Img ? $"img2img (start={startStep}/{steps})"
                       : "txt2img";
        Logs.Info($"Flux ({baseMode}) {opMode}: {width}x{height}, {steps} steps, guidance={guidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text (with cross-generation prompt-embedding cache) ──
        // VALIDATION-PENDING (pre-existing): verify true-CFG vs diffusers FluxPipeline true_cfg. diffusers Flux
        // runs a real unconditional pass only when true_cfg_scale > 1 AND a negative prompt is supplied,
        // combining the two velocity predictions via standard CFG (neg + scale*(pos - neg)) in ADDITION to the
        // model's embedded distilled guidance, which still rides along on BOTH passes.
        bool doTrueCfg = trueCfgScale > 1f && negPromptTokenIdsT5 is not null;

        // Cache keys fold the EOS position in with the token ids (the pooled CLIP vector depends on it).
        int[] condKey = BuildPromptCacheKey(promptTokenIdsL, promptEosPositionL, promptTokenIdsT5);
        int[]? negKey = doTrueCfg
            ? BuildPromptCacheKey(negPromptTokenIdsL ?? promptTokenIdsL, negPromptEosPositionL, negPromptTokenIdsT5!)
            : null;
        bool condHit = _cachedCondKey is not null && condKey.AsSpan().SequenceEqual(_cachedCondKey);
        bool negHit = !doTrueCfg || (_cachedNegKey is not null && negKey!.AsSpan().SequenceEqual(_cachedNegKey));

        Tensor clipPooled;
        Tensor t5Embeddings;
        Tensor? negClipPooled = null;
        Tensor? negT5Embeddings = null;
        if (condHit && negHit)
        {
            Logs.Info("Flux prompt-embedding cache HIT — skipping CLIP+T5 encode.");
            clipPooled = _cachedClipPooled!;
            t5Embeddings = _cachedT5!;
            if (doTrueCfg)
            {
                negClipPooled = _cachedNegClipPooled;
                negT5Embeddings = _cachedNegT5;
            }
        }
        else
        {
            Logs.Info("Encoding text with CLIP-L (pooled) + T5-XXL (per-token)...");

            // The T5 cannot always coexist with a resident DiT (HARTSY_KEEP_MODELS); evict for this
            // encode, the denoise section re-preloads. Cache hits never pay this — and a text encoder placed
            // on its OWN device (TextEncoderBackend) never contends with the DiT at all, so skip the evict.
            if (_ditResident && ReferenceEquals(TextEncoderBackend, Backend))
            {
                FreeResidentTransformer();
            }

            // Preload T5 weights to GPU as a single batch upload. Without this, every
            // matmul/layernorm inside the encoder would do its own cache-miss H2D
            // transfer + immediate free (see CudaBackend.MatMul finally block) — turning
            // text encoding into thousands of ~MB-sized PCIe ping-pongs instead of one
            // bulk transfer + many on-GPU reuses. Backends that don't support a weight
            // cache (Cpu, Vulkan) treat PreloadWeights as a no-op.
            TextEncoderBackend.PreloadWeights(_t5.EnumerateWeights());

            int[][] batchTokenIdsL = [promptTokenIdsL];
            Tensor clipLHidden = _clipL.Encode(TextEncoderBackend, batchTokenIdsL);
            LogTensorStats("CLIP hidden (full)", clipLHidden);
            clipPooled = ExtractEosHiddenState(clipLHidden, promptEosPositionL);
            clipLHidden.Dispose();

            int[][] batchTokenIdsT5 = [promptTokenIdsT5];
            int[][]? batchMaskT5 = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
            t5Embeddings = _t5.Encode(TextEncoderBackend, batchTokenIdsT5, batchMaskT5);

            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (T5 seqLen={t5Embeddings.Shape[1]})");
            LogTensorStats("CLIP pooled", clipPooled);
            LogTensorStats("T5 embeddings", t5Embeddings);

            if (doTrueCfg)
            {
                Logs.Info($"Flux true-CFG enabled (true_cfg_scale={trueCfgScale}); encoding negative prompt...");
                int[][] negBatchTokenIdsL = [negPromptTokenIdsL ?? promptTokenIdsL];
                Tensor negClipLHidden = _clipL.Encode(TextEncoderBackend, negBatchTokenIdsL);
                negClipPooled = ExtractEosHiddenState(negClipLHidden, negPromptEosPositionL);
                negClipLHidden.Dispose();

                int[][] negBatchTokenIdsT5 = [negPromptTokenIdsT5!];
                int[][]? negBatchMaskT5 = negPromptAttentionMaskT5 is not null ? [negPromptAttentionMaskT5] : null;
                negT5Embeddings = _t5.Encode(TextEncoderBackend, negBatchTokenIdsT5, negBatchMaskT5);
                LogTensorStats("Neg CLIP pooled", negClipPooled);
                LogTensorStats("Neg T5 embeddings", negT5Embeddings);
            }

            // Free T5 weights from GPU now that text encoding is done. T5-XXL is ~5 GB —
            // keeping it cached through sampling + VAE decode would OOM 12 GB cards on Flux.
            // The activation tensors live independently of the encoder weights.
            TextEncoderBackend.FreeWeights(_t5.EnumerateWeights());

            // Materialize the conditioning on the host (clipPooled already is — ExtractEosHiddenState reads it),
            // then reclaim every encoder intermediate. CLIP+T5 leave hundreds of device-cached activations that
            // otherwise linger until GC finalization — they held multi-GB into the DiT phase and were a chunk of
            // the fp8 auto-transfer OOM. Host-materializing is also what lets the cached embeddings survive
            // every later FreeActivations sweep across generations.
            // LOAD-BEARING for TextEncoderDevice placement: these host reads ARE the cross-device boundary —
            // the denoiser's backend re-uploads the conditioning from the host copies.
            _ = t5Embeddings.DataPointer;
            if (negT5Embeddings is not null) _ = negT5Embeddings.DataPointer;
            TextEncoderBackend.FreeActivations();

            // Install into the cross-generation cache (dispose whatever it replaces).
            if (!condHit)
            {
                if (_cachedClipPooled != clipPooled) _cachedClipPooled?.Dispose();
                if (_cachedT5 != t5Embeddings) _cachedT5?.Dispose();
                _cachedCondKey = condKey;
                _cachedClipPooled = clipPooled;
                _cachedT5 = t5Embeddings;
            }
            if (doTrueCfg && !negHit)
            {
                if (_cachedNegClipPooled != negClipPooled) _cachedNegClipPooled?.Dispose();
                if (_cachedNegT5 != negT5Embeddings) _cachedNegT5?.Dispose();
                _cachedNegKey = negKey;
                _cachedNegClipPooled = negClipPooled;
                _cachedNegT5 = negT5Embeddings;
            }
        }

        int txtSeqLen = (int)t5Embeddings.Shape[1];
        if (doTrueCfg && (int)negT5Embeddings!.Shape[1] != txtSeqLen)
        {
            // The negative T5 stream must match the positive's sequence length so the transformer's joint
            // [txt|img] attention and the velocity shapes line up for the CFG combine.
            throw new InvalidOperationException(
                $"True-CFG negative T5 seqLen ({negT5Embeddings.Shape[1]}) must equal positive T5 seqLen ({txtSeqLen}). " +
                "Tokenize the negative prompt with the same max length as the positive prompt.");
        }

        // ── 1b. FLUX.1 Redux: append the projected SigLIP image tokens after the T5 text tokens. They ride
        //    the joint attention as extra text tokens with the same all-zero RoPE position (diffusers
        //    FluxPriorRedux / Comfy StyleModelApply). Callers pre-scale the embeds by the style strength;
        //    a non-zero apply-start fraction makes the loop switch conditioning per step (host path).
        Tensor? reduxExtendedT5 = null;
        int reduxTokenCount = 0;
        if (reduxImageEmbeds is not null)
        {
            if (regionalPlan is not null && regionalPlan.Regions.Count > 0)
            {
                throw new InvalidOperationException("Redux style conditioning and regional prompting cannot be combined in one request.");
            }
            if (reduxImageEmbeds.Shape.Rank != 3 || reduxImageEmbeds.Shape[0] != 1
                || reduxImageEmbeds.Shape[2] != t5Embeddings.Shape[2])
            {
                throw new ArgumentException(
                    $"reduxImageEmbeds must be [1, N, {t5Embeddings.Shape[2]}] (Flux text-space tokens); got {reduxImageEmbeds.Shape}.",
                    nameof(reduxImageEmbeds));
            }
            reduxExtendedT5 = ConcatPackedSeqDim(t5Embeddings, reduxImageEmbeds);
            reduxTokenCount = (int)reduxImageEmbeds.Shape[1];
            Logs.Info($"Flux Redux: appended {reduxTokenCount} style tokens after {txtSeqLen} text tokens (applyStart={reduxApplyStartFraction:F2}).");
        }

        // ── 2. Set up dynamic flow-match scheduler ─────────────────────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;
        TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);

        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent ─────────────────────────────
        // T2I: noise scaled by initSigma. Img2img: vaeEncoder.Encode → Pack → AddNoise at sigma[startStep].
        // Masked inpaint: also keep the packed source latent + packed mask alive for per-step blend.
        (Tensor packedLatent, Tensor? packedSourceLatent) =
            BuildInitialPackedLatent(request, scheduler, latentShape, packedShape, latentH, latentW, seed, startStep, keepSourceLatent: isMaskedInpaint);
        Tensor? packedMask = null;
        if (isMaskedInpaint)
        {
            Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
            packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latentH, latentW);
            latentMask.Dispose();
        }

        // ── 3b. FLUX.1 Tools control: VAE-encode the control image once and pack it.
        //    The packed control [1, S, 64] is concatenated to the packed noise [1, S, 64]
        //    along the feature dim every step → [1, S, 128], which the wider x_embedder
        //    consumes. Since the control image is fixed across the schedule we encode it
        //    once and reuse — meaningful win on long schedules.
        Tensor? packedControl = null;
        if (isFillModel)
        {
            // FLUX.1 Fill conditioning (diffusers FluxFillPipeline): per-token features are
            // [noise 64 | masked-image latent 64 | mask 256]. The masked image is source·(1−mask)
            // in [-1,1] (hole → 0 = the VAE's mid-gray), VAE-encoded with the normal shift/scale.
            // The pixel mask maps each latent cell's 8×8 pixel block to 64 channels (c = sy·8+sx),
            // then 2×2-packs to 256. Fixed across the schedule → built once, concatenated per step.
            ImageToImageRequest fillReq = (ImageToImageRequest)request;
            Stopwatch fillSw = Stopwatch.StartNew();
            Tensor maskedImage = MaskPixelsToNeutral(fillReq.SourceImage, maskPixel!);
            Tensor maskedLatent = _vaeEncoder!.Encode(VaeBackend, maskedImage);  // LOAD-BEARING for VaeDevice: PackLatent below is a host loop
            maskedImage.Dispose();
            Tensor packedMaskedLatent = PackLatent(maskedLatent, latentH, latentW);
            maskedLatent.Dispose();
            Tensor mask64 = FillMaskToLatentChannels(maskPixel!, latentH, latentW);
            Tensor packedFillMask = PackLatent(mask64, latentH, latentW);
            mask64.Dispose();
            packedControl = ConcatPackedFeatureDim(packedMaskedLatent, packedFillMask);
            packedMaskedLatent.Dispose();
            packedFillMask.Dispose();
            fillSw.Stop();
            Logs.Info($"Flux Fill conditioning built in {fillSw.ElapsedMilliseconds}ms (masked-latent 64 + packed mask 256).");
        }
        else if (isToolsModel && controlImage is not null)
        {
            if (controlImage.Shape.Rank != 4 || controlImage.Shape[0] != 1 || controlImage.Shape[1] != 3
                || controlImage.Shape[2] != height || controlImage.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"controlImage shape must be [1, 3, {height}, {width}] (matching request); got {controlImage.Shape}.",
                    nameof(controlImage));
            }
            Stopwatch ctrlEncSw = Stopwatch.StartNew();
            Tensor controlLatentUnpacked = _vaeEncoder!.Encode(VaeBackend, controlImage);  // LOAD-BEARING for VaeDevice: PackLatent below is a host loop
            ctrlEncSw.Stop();
            Logs.Info($"Flux Tools control VAE encode done in {ctrlEncSw.ElapsedMilliseconds}ms");
            packedControl = PackLatent(controlLatentUnpacked, latentH, latentW);
            controlLatentUnpacked.Dispose();
        }

        // ── 3c. Flux.1 Kontext reference image: VAE-encode + pack once to [1, refSeq, 64]. Unlike Tools
        //    (channel concat), the reference tokens are APPENDED to the noise tokens along the SEQUENCE every
        //    step (diffusers FluxKontext) and given RoPE temporal-axis = 1 so the model distinguishes them.
        //    The reference is expected at the output resolution, so its packed grid = (hPacked, wPacked).
        Tensor? packedKontextRef = null;
        int kontextRefSeqLen = 0;
        if (kontextRefImage is not null)
        {
            if (_vaeEncoder is null)
                throw new InvalidOperationException("Kontext reference image requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");
            if (kontextRefImage.Shape.Rank != 4 || kontextRefImage.Shape[1] != 3
                || kontextRefImage.Shape[2] != height || kontextRefImage.Shape[3] != width)
                throw new ArgumentException(
                    $"kontextRefImage shape must be [1, 3, {height}, {width}] (resized to the output resolution); got {kontextRefImage.Shape}.",
                    nameof(kontextRefImage));
            Tensor refLatentUnpacked = _vaeEncoder.Encode(VaeBackend, kontextRefImage);
            packedKontextRef = PackLatent(refLatentUnpacked, latentH, latentW);
            refLatentUnpacked.Dispose();
            kontextRefSeqLen = (int)packedKontextRef.Shape[1];
            Logs.Info($"Flux Kontext reference encoded: {kontextRefSeqLen} tokens.");
        }

        // ── 3d. Flux DiT ControlNets: VAE-encode + 2×2-pack each control image ONCE (fixed across the
        //    schedule, like the Tools control). Per step the packed control rides into each adapter's
        //    controlnet_x_embedder — the residuals it produces are what varies with the latent/timestep.
        Tensor[]? cnPackedControls = null;
        if (hasFluxCn)
        {
            cnPackedControls = new Tensor[fluxControlNets!.Count];
            for (int c = 0; c < fluxControlNets.Count; c++)
            {
                Tensor cnImage = fluxControlNets[c].ControlImage;
                if (cnImage.Shape.Rank != 4 || cnImage.Shape[0] != 1 || cnImage.Shape[1] != 3
                    || cnImage.Shape[2] != height || cnImage.Shape[3] != width)
                {
                    throw new ArgumentException(
                        $"Flux ControlNet image {c} shape must be [1, 3, {height}, {width}] (matching request); got {cnImage.Shape}.",
                        nameof(fluxControlNets));
                }
                Tensor cnLatent = _vaeEncoder!.Encode(VaeBackend, cnImage);  // LOAD-BEARING for VaeDevice: PackLatent below is a host loop
                cnPackedControls[c] = PackLatent(cnLatent, latentH, latentW);
                cnLatent.Dispose();
            }
            Logs.Info($"Flux ControlNet: encoded {cnPackedControls.Length} control image(s) to packed latents.");
        }

        // ── 4. Denoising loop ─────────────────────────────────────────
        // Two paths depending on whether the backend can stream:
        //   - StreamingCache != null (CUDA): use BlockStreamingController so resident
        //     VRAM peaks at ~(activations + 2 blocks of weights), making Flux work on
        //     12 GB cards. Shared (non-block) transformer weights still preload eagerly
        //     since they're touched on every step and only ~80 MB total.
        //   - StreamingCache == null (CPU/Vulkan): preload everything eagerly. CPU has
        //     no notion of "device memory"; Vulkan's allocator is independent of this API.
        //     Same behavior as before this refactor.
        // CFG-branch parallelism (ROADMAP.md §1): uncond runs concurrently on a second backend instead of after
        // cond on this one. Structurally blocked, not just untested, when block-streaming is active — the
        // streaming controller's BeforeBlockForward hook is a SINGLE field on the shared _transformer object, so
        // two backends' controllers can't both drive it concurrently. Eligibility can only be decided AFTER the
        // resident-vs-streaming placement below, from the ACTUAL decision for THIS generation (VRAM pressure
        // varies run to run) — never from static config. Falls back to sequential silently (one log line, never
        // a throw) if CfgParallelBackend can't also hold the whole DiT resident (~2× the VRAM of a single card
        // resident, e.g. ~24 GB total for Flux-dev fp8 across two cards — rarely available on consumer pairs;
        // correct when it is, honestly narrow when it isn't).
        bool cfgParallelEligible = false;
        LastCfgParallelDecision = null;
        // Record the no-true-CFG outcome centrally — it holds on EVERY placement branch (resident, streaming,
        // sharded), so recording it inside one branch would leave the others silent, which is exactly the
        // observability gap the diagnostic exists to close.
        if (CfgParallelBackend is not null && !doTrueCfg)
        {
            RecordCfgParallelDecision("inapplicable(no-true-cfg)");
        }
        BlockStreamingController? streamer = null;
        // DiT sharding v1: plain path only (the drainFree feature set). Excluded combinations run unsharded on
        // the primary with a visible log — features take priority over sharding, sharding takes priority over
        // streaming. CFG-parallel + sharding is rejected at config time by PlacementPlanner.ValidatePlacement.
        bool ditShardActive = DitShardBackend is not null && !isMaskedInpaint && packedControl is null
            && packedKontextRef is null && !hasFluxCn && (regionalPlan is null || regionalPlan.Regions.Count == 0)
            && !StatsEnabled && (reduxExtendedT5 is null || reduxApplyStartFraction <= 0f);
        if (DitShardBackend is not null && !ditShardActive)
        {
            Logs.Warning("Flux DiT sharding configured but this generation uses features outside the sharded v1 "
                + "surface (ControlNet/Kontext/inpaint/regional/Redux-start/stats) — running unsharded on the primary backend.");
            if (_ditShardResident)
            {
                FreeResidentTransformer();
            }
        }
        if (ditShardActive)
        {
            if (Backend.StreamingCache is not null)
                Logs.Info("Flux: DiT sharding overrides low-VRAM block streaming for this generation.");
            if (_ditResident && !_ditShardResident)
            {
                // A previous unsharded generation left the whole DiT on the primary — start clean.
                FreeResidentTransformer();
            }
            if (!_ditResident)
            {
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                Backend.PreloadWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
                DitShardBackend!.PreloadWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
            }
            _ditResident = true;
            _ditShardResident = true;
        }
        else if (Backend.StreamingCache is not null)
        {
            IStreamingBlock[] blocks = new IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);

            // Kontext appends refSeqLen reference tokens to the noise tokens, so the transformer's real
            // per-forward activation working set spans txt + noise + ref — not just the noise seq. Reserve for
            // the full length so the estimates don't under-count and over-commit VRAM (base Flux
            // passes kontextRefSeqLen = 0, so this is unchanged there).
            int forwardImgSeqLen = imgSeqLen + kontextRefSeqLen;
            long activationReserve = EstimateFluxActivationReserveBytes(
                txtSeqLen, forwardImgSeqLen, _config.HiddenSize, (int)(_config.HiddenSize * _config.MlpRatio));

            // RESIDENT fast path: when the whole block set fits beside the activation reserve, skip streaming
            // entirely. The sliding window (retainBehind: 0) evicts and re-uploads EVERY block on EVERY
            // forward — ~12 GB of PCIe traffic per step for Flux-Dev fp8 — which dominated the wall on cards
            // that never needed streaming in the first place (Krea2, same weight class, always ran resident).
            long totalBlockBytes = 0;
            foreach (IStreamingBlock block in blocks) totalBlockBytes += block.EstimatedWeightBytes;
            // The resident-vs-streamed decision (including the already-resident short-circuit that keeps warm
            // generations from oscillating resident→streaming→resident) now lives in VramPlanner, so every
            // pipeline makes it the same way and HARTSY_LOWVRAM can override it. On the default `auto` policy
            // this is exactly the comparison that was inlined here.
            VramPlanner planner = new VramPlanner(Backend.StreamingCache, "Flux", Backend);
            PhasePlacement placement = planner.PlanPhase(
                "denoise", totalBlockBytes, activationReserve, alreadyResident: _ditResident, canStream: true);
            if (placement == PhasePlacement.Resident)
            {
                Backend.PreloadWeights(_transformer.EnumerateWeights());
                cfgParallelEligible = TryPreloadCfgParallel(doTrueCfg);
            }
            else
            {
                if (doTrueCfg && CfgParallelBackend is not null)
                {
                    RecordCfgParallelDecision("fell-back(block-streaming-active)");
                }
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                int prefetchAhead = ChooseFluxPrefetchAhead(Backend.StreamingCache, blocks, activationReserve);
                streamer = new BlockStreamingController(
                    Backend.StreamingCache, blocks, prefetchAhead: prefetchAhead, retainBehind: 0, backend: Backend);
                _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
                streamer.Prime();
                long perBlockMb = streamer.EstimatedTotalWeightBytes / blocks.Length / (1024 * 1024);
                long totalMb = streamer.EstimatedTotalWeightBytes / (1024 * 1024);
                Logs.Info($"Flux streaming: {blocks.Length} blocks, prefetchAhead={prefetchAhead}, " +
                    $"per-block ~{perBlockMb} MB, total ~{totalMb} MB");
            }
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            cfgParallelEligible = TryPreloadCfgParallel(doTrueCfg);
        }

        // ControlNet adapter weights ride beside the DiT for the whole loop (they run every step).
        if (hasFluxCn)
        {
            foreach (Adapters.FluxControlNetConditioning cn in fluxControlNets!)
                Backend.PreloadWeights(cn.Adapter.EnumerateWeights());
        }

        // ── 4b. Regional conditioning: append region T5 streams + prepare per-step attention bias.
        //    Image tokens follow the text tokens in Flux's joint [txt|img] attention, so the bias
        //    rectangle starts at the (possibly extended) text length. Collapses to the base path
        //    when no plan is given.
        bool hasRegions = regionalPlan is not null && regionalPlan.Regions.Count > 0;
        Tensor condStream = reduxExtendedT5 ?? t5Embeddings;
        int condTxtSeqLen = txtSeqLen + reduxTokenCount;
        Tensor? extendedT5 = null;
        List<(int Start, int End)>? regionRanges = null;
        List<float[]>? regionGridMasks = null;
        float[]? regionWeights = null;
        if (hasRegions)
        {
            (extendedT5, condTxtSeqLen, regionRanges, regionGridMasks) =
                RegionalConditioningLayout.BuildTextStream(regionalPlan!, t5Embeddings, hPacked, wPacked);
            condStream = extendedT5;
            regionWeights = new float[regionalPlan!.Regions.Count];
        }

        // Drain-free / graph routing decided BEFORE the pre-loop sweep: FreeActivations resets the backend
        // step-graph slot, so a persistent cross-generation graph requires skipping the sweep on the plain
        // t2i cache-warm path (the Chroma pattern — per-op disposal + the warm mem-pool keep VRAM flat).
        bool drainFree = !isMaskedInpaint && packedControl is null && packedKontextRef is null
            && !hasFluxCn && !hasRegions && !StatsEnabled
            && (reduxExtendedT5 is null || reduxApplyStartFraction <= 0f);

        // Across-step First-Block cache (same knobs as Sd3/Chroma/HiDream). Only wired onto the drainFree,
        // non-sharded path — ForwardSharded has no cache-consuming entry point, and the host-step branch
        // (masked inpaint, ControlNet, Kontext, regional, Redux-mid-stream, stats) is exactly the feature set
        // the transformer's own cacheActive check already excludes (attnBias/controlNetResiduals/refSeqLen>0),
        // so wiring it there would be a silent no-op anyway. Deliberately NOT wired into the CFG-parallel
        // branch below — that already runs cond/uncond concurrently on two backends, and layering a second
        // form of per-step-variable behavior on top of that concurrency needs its own dedicated verification
        // pass, not bundled in here; when a cache is armed alongside CFG-parallel, the parallel branch just
        // runs uncached (silent, not incorrect — Reuses stays 0 for that generation).
        bool stepCacheFastPath = drainFree && !ditShardActive;
        (float stepCacheThreshold, int stepCacheCap, float[]? stepCachePoly, float stepCacheLate) = StepCacheEnv.Resolve(null);
        DeviceFeatureCache? stepCacheCond = null;
        DeviceFeatureCache? stepCacheUncond = null;
        if (stepCacheThreshold > 0f && stepCacheFastPath)
        {
            if (Backend.SupportsDeviceStepCacheGate)
            {
                stepCacheCond = new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile());
                stepCacheUncond = doTrueCfg ? new DeviceFeatureCache(stepCacheThreshold, stepCacheCap, stepCachePoly, StepCacheEnv.ReadCalibFile()) : null;
                Logs.Info($"Step cache ON: threshold={stepCacheThreshold}, maxConsecutiveReuse={stepCacheCap}"
                    + (stepCachePoly is not null ? ", poly gate" : "")
                    + (stepCacheLate > 0f ? $", lateWindow={stepCacheLate}" : ""));
            }
            else
            {
                Logs.Warning("HARTSY_STEP_CACHE set but the backend lacks a device-side gate " +
                    "(stepcache.ptx not compiled?) — running uncached.");
            }
        }

        // Sampler selection (2026-08-20). Flux.1 is flow-matching, so it had no user-selectable sampler before
        // this. A non-default sampler narrows the step graph for the same reason true-CFG, sharding and the step
        // cache already do: the capture bakes one op sequence, and a second-order sampler adds a forward at an
        // intermediate sigma plus scratch arithmetic between them.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Flux.1",
            startsFromNoisedInit: startStep > 0);
        bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);

        // The host/reference branch (masked inpaint, ControlNet, Kontext, regional, Redux-mid-stream, stats)
        // never consults the sampler, so a non-default selection there would be silently dropped.
        if (!drainFree && nonDefaultSampler)
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' runs only on Flux.1's drain-free path, and this generation "
                + "fell back to the reference loop (masked inpaint, ControlNet, Kontext, regional prompting, "
                + "Redux mid-stream, or stats). Drop the sampler selection, or drop the feature that forced the "
                + "fallback.");
        }

        bool graphRoute = drainFree && !doTrueCfg && packedSourceLatent is null && !ditShardActive
            && stepCacheCond is null && !nonDefaultSampler
            && DitStepGraph.EnabledDefaultOn && Backend.StepGraphSupported;

        // Mirrors the per-step dispatch condition (loop-invariant), recorded once per generation so operators
        // and tests can see which CFG-parallel path this generation actually took. The eligible=false cases
        // (preload failure, block streaming) were already recorded where they were decided.
        if (doTrueCfg && cfgParallelEligible)
        {
            if (!drainFree)
            {
                RecordCfgParallelDecision("fell-back(eager-path-features)");
            }
            else if (condTxtSeqLen != txtSeqLen)
            {
                RecordCfgParallelDecision("fell-back(rope-signature-mismatch)");
            }
            else
            {
                // Worker threads must never be the first reader of a promoted host tensor: a prior
                // sequential-CFG generation can auto-promote the cached negatives into the primary's weight
                // cache, and the demote hook must fire HERE, not from the worker mid-cond-forward.
                _ = negT5Embeddings!.DataPointer;
                _ = negClipPooled!.DataPointer;
                RecordCfgParallelDecision("active");
            }
        }

        // Materialize every tensor that must survive across steps on the host, then reclaim the VAE-encode /
        // packing intermediates (control, Kontext, img2img source). The per-step FreeActivations below frees
        // device buffers WITHOUT a D2H sync-back, so anything still device-only here would be silently lost.
        _ = condStream.DataPointer;
        _ = packedLatent.DataPointer;
        if (packedControl is not null) _ = packedControl.DataPointer;
        if (cnPackedControls is not null)
        {
            foreach (Tensor cnPacked in cnPackedControls) _ = cnPacked.DataPointer;
        }
        if (packedKontextRef is not null) _ = packedKontextRef.DataPointer;
        if (packedSourceLatent is not null) _ = packedSourceLatent.DataPointer;
        if (packedMask is not null) _ = packedMask.DataPointer;
        if (!graphRoute)
            Backend.FreeActivations();

        Logs.Info("Starting Flux denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Drain-free fast path (plain t2i / img2img, the Chroma/Krea2 pattern): the latent stays
        // device-resident across the whole loop and the (true-)CFG combine + Euler step run as ONE in-place
        // device op (CfgEulerStep: v = g·pos + (1−g)·neg, then z += v·dt; g=1 degenerates to plain Euler) —
        // no per-step D2H of the velocity + H2D re-upload of the latent. Tools/Kontext (host-side per-step
        // concats), regional bias, masked inpaint (host blend), and the opt-in stats scans keep the host
        // branch, which is unchanged.
        Logs.Info(drainFree ? "Flux loop: drain-free device-resident path." : "Flux loop: host-step path.");

        // Persistent step graph (the Chroma round-3 recipe): route the latent through the transformer-owned
        // FIXED buffer so the whole forward is captured once per (prompt refs, grid) signature and replayed
        // with one cuGraphLaunch per step — it survives across generations (KEEP_MODELS + drain-free +
        // sweep-skip keep the buffers alive, so a repeat-prompt gen replays every step with zero re-capture).
        // True-CFG stays eager (pair capture not wired; Dev/Schnell don't use it).
        if (graphRoute)
        {
            Tensor fixedLatent = _transformer.PrepareGraphLatent(Backend, packedLatent);
            packedLatent.Dispose();
            packedLatent = fixedLatent;
        }

        float[] timestepTable = timesteps.ToArray();

        // The drain-free step body, lifted behind IDenoisePredictor. Everything Flux-specific stays here —
        // CFG-branch parallelism across two backends, DiT sharding, the step cache — and the sampler sees only
        // (x, sigma) -> prediction pair.
        DenoisePrediction PredictDrainFree(Tensor x, float s, int stepIndex)
        {
            // On-schedule sigmas reuse the loop's own `timesteps[i]/1000` expression. The two are equal
            // mathematically but the F32 round trip through x1000 is not exact, so passing raw sigma would shift
            // every existing Flux generation by an ulp of conditioning. A genuine sub-step takes the raw value.
            float stepSigma = stepIndex < steps && s == scheduler.SigmaAt(stepIndex)
                ? timestepTable[stepIndex] / 1000.0f
                : s;
            // Narrowed for a non-default sampler: the step cache is calibrated on drift between CONSECUTIVE
            // steps, and a second-order method evaluates twice inside one step.
            bool eligible = !nonDefaultSampler
                && (stepCacheLate <= 0f || (stepIndex + 1) > steps * (1f - stepCacheLate));

            // condTxtSeqLen != txtSeqLen (Redux tokens / regional conditioning extend the cond stream only)
            // means cond and uncond would precompute DIFFERENT RoPE signatures on the SAME shared _rope
            // object. FluxRope.Precompute is now lock-safe against concurrent calls, but a lock only stops
            // torn writes — it cannot make one cached _cosCache serve two different signatures at once, so
            // running cond/uncond concurrently here would still let one branch's tables get clobbered by the
            // other's mid-step. Fall back to sequential (silent, matches the rest of this eligibility chain)
            // whenever the signatures could actually differ.
            if (doTrueCfg && cfgParallelEligible && condTxtSeqLen == txtSeqLen)
            {
                // CopyFromPeer, not .DataPointer, for BOTH hops — that's what keeps drainFree drain-free.
                // The latent stays device-resident and cache-hit on Backend throughout (CopyFromPeer reads
                // the raw device pointer via TryGetDevicePointer, never touching .DataPointer, so it can't
                // trip the demote hook that would otherwise evict Backend's cached copy from under its own
                // concurrent kernel launches — the same hazard CfgBranchRunner's doc comment calls out for
                // shared weight tensors, but here for a per-step activation instead). uncondLatent is an
                // independent tensor only the worker thread ever touches.
                Tensor uncondLatent = new Tensor(x.Shape, x.DType);
                CfgParallelBackend!.CopyFromPeer(uncondLatent, x, Backend);
                try
                {
                    (Tensor cond, Tensor velocityNegRemote) = CfgBranchRunner.Run(
                        () => _transformer.Forward(Backend, x, condStream, stepSigma,
                            clipPooled!, guidanceScale, condTxtSeqLen, hPacked, wPacked, null, 0, 0, 0),
                        () => _transformer.Forward(CfgParallelBackend!, uncondLatent, negT5Embeddings!, stepSigma,
                            negClipPooled!, guidanceScale, txtSeqLen, hPacked, wPacked, null, 0, 0, 0));
                    Tensor velocityNegLocal = new Tensor(velocityNegRemote.Shape, velocityNegRemote.DType);
                    Backend.CopyFromPeer(velocityNegLocal, velocityNegRemote, CfgParallelBackend!);
                    velocityNegRemote.Dispose();
                    return new DenoisePrediction(cond, velocityNegLocal, trueCfgScale);
                }
                finally { uncondLatent.Dispose(); }
            }

            Tensor velocityPred = RunPlainForward(ditShardActive, x, condStream, stepSigma,
                clipPooled!, guidanceScale, condTxtSeqLen, hPacked, wPacked,
                eligible ? stepCacheCond : null);
            if (!doTrueCfg)
            {
                return new DenoisePrediction(velocityPred, velocityPred);
            }
            Tensor velocityNeg = RunPlainForward(ditShardActive, x, negT5Embeddings!, stepSigma,
                negClipPooled!, guidanceScale, txtSeqLen, hPacked, wPacked,
                eligible ? stepCacheUncond : null);
            return new DenoisePrediction(velocityPred, velocityNeg, trueCfgScale);
        }

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f; // Convert timestep back to sigma [0,1]
            bool cacheEligible = stepCacheLate <= 0f || (i + 1) > steps * (1f - stepCacheLate);

            if (graphRoute)
            {
                (Tensor velocity, bool callerOwns) = _transformer.ForwardGraphable(
                    Backend, packedLatent, condStream, sigma,
                    clipPooled!, guidanceScale, condTxtSeqLen, hPacked, wPacked);
                // Euler stays OUTSIDE the capture: in-place on the fixed latent the graph reads next step.
                Backend.CfgEulerStep(packedLatent, velocity, velocity, 1.0f, scheduler.Dt(i));
                if (callerOwns) velocity.Dispose();
            }
            else if (drainFree)
            {
                DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                    PredictionType.FlowVelocity,
                    (x, s, stepIndex) => PredictDrainFree(x, s, stepIndex));
                if (i == startStep)
                {
                    sampler.Reset(packedLatent.Shape);
                }
                sampler.Step(Backend, packedLatent, predictor, i);
            }
            else
            {
            // For FLUX.1 Tools: concat noise + control along the feature dim before the
            // transformer pass. The transformer's wider x_embedder consumes the 128-dim
            // input; velocity output is back at 64 dims (no control side in the prediction).
            Tensor transformerInput = packedControl is not null
                ? ConcatPackedFeatureDim(packedLatent, packedControl)      // Tools/Fill: channel concat → 128/384-dim
                : packedKontextRef is not null
                    ? ConcatPackedSeqDim(packedLatent, packedKontextRef)   // Kontext: sequence concat [noise;ref]
                    : packedLatent;

            // Redux apply-start: before the threshold step the forward runs on the plain text stream,
            // from it onward on the redux-extended stream (Comfy ConditioningSetTimestepRange semantics).
            bool reduxActive = reduxExtendedT5 is null || reduxApplyStartFraction <= 0f
                || (i - startStep) >= (int)MathF.Ceiling(reduxApplyStartFraction * (steps - startStep));
            Tensor stepCond = reduxActive ? condStream : t5Embeddings;
            int stepCondLen = reduxActive ? condTxtSeqLen : txtSeqLen;

            Tensor? regionBias = null;
            if (hasRegions)
            {
                regionalPlan!.ResolveStep(i - startStep, regionWeights!);
                regionBias = RegionalAttentionBias.Build(
                    condTxtSeqLen + imgSeqLen, condTxtSeqLen, imgSeqLen, regionRanges!, regionGridMasks!, regionWeights!);
            }

            // Flux DiT ControlNets: run every active adapter against the current latent + its packed control
            // and sum the residual stacks (diffusers FluxMultiControlNetModel union stacking). The residuals
            // feed both the conditional and (for true-CFG) the unconditional transformer pass this step.
            Adapters.FluxControlNetResiduals? cnResiduals = null;
            if (hasFluxCn)
            {
                for (int c = 0; c < fluxControlNets!.Count; c++)
                {
                    Adapters.FluxControlNetConditioning cn = fluxControlNets[c];
                    if (!cn.IsActiveAtStep(i, steps)) continue;
                    Adapters.FluxControlNetResiduals next = cn.Adapter.Forward(
                        Backend, packedLatent, cnPackedControls![c], sigma, t5Embeddings, clipPooled!,
                        guidanceScale, hPacked, wPacked,
                        cn.UnionMode is FluxUnionControlMode unionMode ? (int)unionMode : null, cn.Scale);
                    cnResiduals = cnResiduals is null ? next : SumControlNetResiduals(cnResiduals, next);
                }
            }

            // Forward pass: conditional velocity prediction (embedded distilled guidance rides along).
            Tensor velocityPred = _transformer.Forward(
                Backend, transformerInput, stepCond, sigma,
                clipPooled!, guidanceScale, stepCondLen, hPacked, wPacked, regionBias,
                kontextRefSeqLen, kontextRefSeqLen > 0 ? hPacked : 0, kontextRefSeqLen > 0 ? wPacked : 0,
                cnResiduals);

            // True-CFG: run a second forward with the negative conditioning (same noisy latent,
            // same timestep, same embedded guidanceScale) and combine with standard CFG. The
            // negative pass uses the base T5 length (no regional extension); regional plans
            // extend only the positive stream, so true-CFG runs against the unextended negative.
            // VALIDATION-PENDING: verify vs diffusers FluxPipeline true_cfg.
            if (doTrueCfg)
            {
                Tensor velocityNeg = _transformer.Forward(
                    Backend, transformerInput, negT5Embeddings!, sigma,
                    negClipPooled!, guidanceScale, txtSeqLen, hPacked, wPacked, null,
                    kontextRefSeqLen, kontextRefSeqLen > 0 ? hPacked : 0, kontextRefSeqLen > 0 ? wPacked : 0,
                    cnResiduals);
                Tensor combined = CfgHelper.ApplyCfg(velocityNeg, velocityPred, trueCfgScale);
                velocityNeg.Dispose();
                velocityPred.Dispose();
                velocityPred = combined;
            }

            cnResiduals?.DisposeAll();
            regionBias?.Dispose();
            if (transformerInput != packedLatent) transformerInput.Dispose();

            LogTensorStats($"Step {i+1} velocity", velocityPred);
            LogPerLatentChannelMeanPacked($"Step {i+1} velocity", velocityPred);

            // Scheduler step: Euler on packed latent
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, 64);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            SchedulerStepPacked(newLatent, velocityPred, packedLatent, scheduler, i);
            velocityPred.Dispose();
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
                    Tensor freshPackedNoise = PackLatent(freshUnpackedNoise, latentH, latentW);
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

            LogPerLatentChannelMeanPacked($"Step {i+1} latent", packedLatent);

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            // packedLatent is [B, S, 64] (16 channels × 2×2 patches per token). LatentPreview
            // needs unpacked NCHW — allocate a temp, emit, dispose. On the drain-free path the unpack reads
            // the device-resident latent (a D2H sync per frame), so previews emit every 4th step + final
            // instead of every step; the host path keeps every step.
            bool emitPreview = onProgress is not null && (!drainFree || (i - startStep) % 4 == 3 || i == steps - 1);
            if (emitPreview)
            {
                // Graph route: packedLatent is the transformer-owned FIXED buffer whose device address is
                // baked into the captured step graph. A host DataPointer read (which the unpack does) would
                // D2H-and-FREE that buffer via the activation sync callback — every later replay then reads
                // freed, reused pool memory (warm-gen noise; whether it shows is pool-reuse luck, which is
                // why 512² passed while 1024² speckled). Preview from an address-preserving device snapshot
                // instead — the same rule Chroma's preview hook already follows.
                Tensor previewSource = graphRoute ? _transformer.SnapshotGraphLatent(Backend) : packedLatent;
                try
                {
                    Tensor previewLatent = LatentPreview.UnpackFluxStylePacked(previewSource, latentH, latentW, channels: 16);
                    try
                    {
                        onProgress!.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                        {
                            Latent = previewLatent,
                            LatentArch = LatentArchitecture.Flux,
                        });
                    }
                    finally { previewLatent.Dispose(); }
                }
                finally
                {
                    if (!ReferenceEquals(previewSource, packedLatent)) previewSource.Dispose();
                }
            }
            else if (onProgress is not null)
            {
                onProgress.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
            }

            // Reclaim GPU-resident activation buffers between steps on the HOST path only: there the next
            // step's packedLatent is host-resident (SchedulerStepPacked runs on the host), so freeing device
            // buffers is safe. On the drain-free path the latent LIVES in the activation cache — freeing it
            // would silently revert the next step to the stale pre-loop host copy (the Wan I2V garbage bug);
            // per-op disposal + the warm mem-pool keep drain-free VRAM flat without a sweep (Chroma-verified).
            if (!drainFree)
            {
                Backend.FreeActivations();
            }
            // CfgParallelBackend never holds packedLatent (only the per-step uncondLatent clone, already disposed
            // above) — nothing to protect there, and its per-step DiT-internal activations need a sweep or they
            // accumulate to OOM over the loop exactly like Wan's uncond branch does (CfgBranchRunner's doc
            // comment). Unlike Backend, this is unconditional even on drainFree.
            if (doTrueCfg && cfgParallelEligible)
            {
                CfgParallelBackend!.FreeActivations();
            }
        }

        // On the graph route packedLatent IS the transformer-owned fixed buffer (alive across generations
        // for graph replay) — hand a snapshot to the unpack/dispose tail instead.
        if (graphRoute)
            packedLatent = _transformer.SnapshotGraphLatent(Backend);

        if (stepCacheCond is not null)
        {
            string uncondStats = stepCacheUncond is not null
                ? $"; uncond {stepCacheUncond.Computes} computes / {stepCacheUncond.Reuses} reuses"
                : "";
            Logs.Info($"Step cache: cond {stepCacheCond.Computes} computes / {stepCacheCond.Reuses} reuses{uncondStats}");
        }
        stepCacheCond?.Dispose();
        stepCacheUncond?.Dispose();

        // clipPooled / t5Embeddings / negClipPooled / negT5Embeddings are cross-generation cache entries —
        // not disposed here; replacement on a future cache miss owns their lifetime.
        extendedT5?.Dispose();
        reduxExtendedT5?.Dispose();
        packedSourceLatent?.Dispose();
        packedMask?.Dispose();
        packedControl?.Dispose();
        packedKontextRef?.Dispose();
        if (cnPackedControls is not null)
        {
            foreach (Tensor cnPacked in cnPackedControls) cnPacked.Dispose();
        }
        if (hasFluxCn)
        {
            // ControlNet weights always evict after the loop (adapters are per-request conditioning, not a
            // resident model) — frees room for the VAE decode.
            foreach (Adapters.FluxControlNetConditioning cn in fluxControlNets!)
                Backend.FreeWeights(cn.Adapter.EnumerateWeights());
        }

        // Tear down the streaming controller (frees still-resident blocks) and free the
        // shared weights, making room for VAE decode on tight VRAM budgets. On the eager path under
        // HARTSY_KEEP_MODELS the DiT stays resident across generations instead (the full-res VAE decode's
        // banded im2col fits beside it, falling back to tiles if not — the Chroma pattern); a future
        // prompt-cache miss evicts it before the T5 encode.
        _transformer.BeforeBlockForward = null;
        if (streamer is not null)
        {
            streamer.EvictAll();
            streamer.Dispose();
            _transformer.InvalidateStepGraph(Backend);
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            // Also purge the block weights: the streaming cache registers each uploaded block in the backend
            // weight cache, and any block still cached at the end (or its lingering F16 cast) would stay
            // resident and starve the VAE decode (same fix as HunyuanVideoPipeline — decode OOM'd with the
            // DiT still holding ~20+ GB, <1.2 GB free).
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        else if (!KeepModelsResident)
        {
            FreeResidentTransformer();
        }
        else
        {
            _ditResident = true;
        }

        // ── 5. Unpack latent: [1, seqLen, 64] → [1, 16, latentH, latentW] ──
        LogTensorStats("Final packed latent", packedLatent);
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();
        LogTensorStats("Unpacked latent", unpackedLatent);
        LogPerChannelStats("Unpacked latent", unpackedLatent);

        // Save unpacked latent for Python cross-validation if output dir exists
        string? debugLatentDir = Environment.GetEnvironmentVariable("FLUX_DEBUG_DIR");
        if (debugLatentDir is not null)
        {
            Directory.CreateDirectory(debugLatentDir);
            string latentPath = Path.Combine(debugLatentDir, "unpacked_latent.bin");
            ReadOnlySpan<float> latentData = unpackedLatent.AsReadOnlySpan<float>();
            using FileStream fs = new FileStream(latentPath, FileMode.Create);
            using BinaryWriter bw = new BinaryWriter(fs);
            for (int i = 0; i < latentData.Length; i++)
                bw.Write(latentData[i]);
            Logs.Info($"Saved unpacked latent to {latentPath} ({latentData.Length} floats, shape={unpackedLatent.Shape})");
        }

        // ── 6. VAE decode ─────────────────────────────────────────────
        // Preload VAE weights too — many small conv+norm ops in a row, same per-op
        // re-upload pattern would apply. Runs on VaeBackend (defaults to Backend): the unpacked latent above is
        // host-side (UnpackLatent), so a VAE placed on another device just uploads from there.
        VaeBackend.PreloadWeights(_vaeDecoder.EnumerateWeights());

        // Always go through DecodeTiled. It internally fast-paths to a single direct
        // decode when the latent fits in one tile (small images), so there's no overhead
        // for normal sizes — but for anything ≥1024² it slices into 64-latent / 512-RGB
        // tiles with a 64-pixel RGB blend overlap. This kills the catastrophic im2col
        // workspace blow-up: a 256ch 3×3 conv at 1024² needs ~9.7 GB workspace as one
        // shot vs ~2.4 GB per tile (F32), regardless of final resolution.
        // (Tiles run at F32; F16 VAE produced black output on Flux Schnell — needs
        // GroupNorm/softmax precision investigation before re-enabling.)
        Logs.Info("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(VaeBackend, unpackedLatent);
        unpackedLatent.Dispose();
        LogTensorStats("VAE output", image);
        LogPerChannelStats("VAE output", image);
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 7. Pixel-space recomposite for masked inpaint ─────────────
        // Paste decoded pixels over source pixels where mask=1; suppresses VAE encode/
        // decode drift in unmasked regions. Identical math to SDXL — Flux's 16ch latent
        // affects only the per-step blend, not the final RGB recomposite.
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // ── 8. Convert to RGB bytes ───────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        // Final reclaim: in a long-lived host (SwarmUI), VAE-decode intermediates otherwise sit in device
        // memory until GC finalization and shrink the budget of whatever generation runs next. While a
        // captured step graph is alive, FreeActivations would destroy it (and free its fixed buffers) —
        // trim the pool instead (sync + return-to-driver; per-op disposal already freed the intermediates).
        if (Backend.StepGraphReady)
            Backend.TrimMemoryPool();
        else
            Backend.FreeActivations();

        sw.Stop();
        Logs.Info($"Flux ({baseMode}) {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Picks <c>prefetchAhead</c> for the streaming controller based on how much VRAM is left after reserving for the peak activation working set of a single forward pass. Each extra prefetched block costs one block's worth of resident weights in addition to the current+next block already in flight. We pick the largest value that still leaves headroom for activations + cuBLAS workspace, capped at 2 (deeper just churns VRAM without extra hiding when blocks compute in tens of milliseconds).</summary>
    /// <summary>Estimates the peak per-forward activation footprint (bytes) that must stay free beside the DiT weights — the deepest valley is a SingleStreamBlock holding the F16 mlpInput + mlpActivated and the F16 concatted simultaneously alongside the F32 attention tensors. Shared by the resident-vs-stream decision and the prefetch-depth choice so the two can never disagree. Byte sizes are for B=1.</summary>
    /// <summary>Attempts to preload the whole DiT onto <see cref="DiffusionPipelineBase.CfgParallelBackend"/> so the true-CFG uncond branch can run there concurrently with cond on <see cref="DiffusionPipelineBase.Backend"/>. Never throws — a card that can't also hold the DiT resident (~2× a single card's worth of VRAM) falls back to the sequential path with one log line, exactly like the "no CfgParallelBackend configured" case. Called only from the Resident placement branches; block-streaming and CFG-parallel don't compose (see the call sites' comment).</summary>
    /// <summary>Routes a plain-path forward (no attnBias/Kontext/ControlNet, the only calls DiT sharding v1 supports) to <see cref="FluxTransformer.ForwardSharded"/> when this generation's sharding is active.</summary>
    /// <summary><paramref name="stepCache"/> only applies on the non-sharded path — <see cref="FluxTransformer.ForwardSharded"/> has no cache-consuming entry point.</summary>
    private Tensor RunPlainForward(bool ditShardActive, Tensor packedInput, Tensor condStream, float sigma,
        Tensor pooled, float guidanceScale, int txtLen, int hPacked, int wPacked, DeviceFeatureCache? stepCache = null) =>
        ditShardActive
            ? _transformer.ForwardSharded(Backend, DitShardBackend!, packedInput, condStream, sigma,
                pooled, guidanceScale, txtLen, hPacked, wPacked, DitShardSplitBlock)
            : _transformer.Forward(Backend, packedInput, condStream, sigma,
                pooled, guidanceScale, txtLen, hPacked, wPacked, null, 0, 0, 0, stepCache: stepCache);

    /// <summary>Frees the resident DiT on whichever backend(s) hold it — the whole set on the primary, or the asymmetric sharded split. The unsharded free would silently no-op on the shard backend's range.</summary>
    private void FreeResidentTransformer()
    {
        _transformer.InvalidateStepGraph(Backend);
        if (_ditShardResident)
        {
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
            Backend.FreeWeights(_transformer.EnumerateBlockRangeWeights(0, DitShardSplitBlock));
            DitShardBackend!.FreeWeights(_transformer.EnumerateBlockRangeWeights(DitShardSplitBlock, _transformer.BlockCount));
            _ditShardResident = false;
        }
        else
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
        }
        _ditResident = false;
    }

    /// <summary>Releases the prompt-embedding cache (pipeline-internal state; see DiffusionPipelineBase).</summary>
    protected override void DisposeCore()
    {
        _cachedClipPooled?.Dispose();
        _cachedClipPooled = null;
        _cachedT5?.Dispose();
        _cachedT5 = null;
        _cachedCondKey = null;
        _cachedNegClipPooled?.Dispose();
        _cachedNegClipPooled = null;
        _cachedNegT5?.Dispose();
        _cachedNegT5 = null;
        _cachedNegKey = null;
    }

    private bool TryPreloadCfgParallel(bool doTrueCfg)
    {
        if (!doTrueCfg || CfgParallelBackend is null)
        {
            // The no-true-CFG outcome is recorded centrally before the placement branches.
            return false;
        }
        try
        {
            CfgParallelBackend.PreloadWeights(_transformer.EnumerateWeights());
            return true;
        }
        catch (Exception ex)
        {
            Logs.Warning($"Flux CFG-parallel: couldn't preload the DiT onto the second backend (falling back to "
                + $"sequential true-CFG this generation): {ex.Message}");
            RecordCfgParallelDecision($"fell-back(preload-failed: {ex.Message})");
            return false;
        }
    }

    private static long EstimateFluxActivationReserveBytes(int txtSeqLen, int imgSeqLen, int hiddenSize, int mlpDim)
    {
        long totalSeqLen = txtSeqLen + imgSeqLen;
        long mlpInputBytes = totalSeqLen * mlpDim * 2;             // F16
        long mlpActivatedBytes = totalSeqLen * mlpDim * 2;         // F16
        long concattedBytes = totalSeqLen * (hiddenSize + mlpDim) * 2; // F16
        long attnFlatBytes = totalSeqLen * hiddenSize * 4;         // F32
        long qkvBytes = 3L * totalSeqLen * hiddenSize * 4;         // F32 q/k/v
        long mhBytes = 3L * totalSeqLen * hiddenSize * 4;          // F32 multi-head q/k/v
        long modulatedBytes = totalSeqLen * hiddenSize * 4;        // F32 modulated/normed
        long xBytes = totalSeqLen * hiddenSize * 4;                // F32 block input
        long scratchBytes = mlpInputBytes + mlpActivatedBytes + concattedBytes
            + attnFlatBytes + qkvBytes + mhBytes + modulatedBytes + xBytes;
        // Add a generous fudge for cuBLAS workspace, fp8→F16 weight cast buffers, RoPE tables,
        // and the persistent shared weights that aren't part of the streamed set. These are
        // hard to predict tightly; 1 GB is roomy enough to keep us out of OOM territory and
        // cheap if we have it.
        return scratchBytes + 1024L * 1024 * 1024;
    }

    private static int ChooseFluxPrefetchAhead(
        IStreamingWeightCache cache,
        IStreamingBlock[] blocks,
        long activationReserve)
    {
        long avail = cache.QueryAvailableWeightCacheBytes(activationReserve);
        if (avail <= 0) return 0;

        long perBlockBytes = blocks.Length > 0 ? blocks[0].EstimatedWeightBytes : 0;
        if (perBlockBytes <= 0) return 1;

        // The streaming working set briefly hits (prefetchAhead + 2) blocks at the moment we
        // make block N+1 resident before evicting block N-1 (see BlockStreamingController.
        // BeforeBlockForward). Pick the largest prefetch that keeps that peak under budget.
        // Cap at 2 — beyond that we burn VRAM without much extra latency hiding.
        int maxByBudget = (int)(avail / perBlockBytes) - 2;
        int chosen = Math.Clamp(maxByBudget, 0, 2);
        return chosen;
    }

    /// <summary>Builds the initial packed latent for Flux denoising. T2I: fresh Gaussian noise scaled by the scheduler's initial sigma. Img2img: VAE-encoded source latent (16 channels) packed via <see cref="PackLatent"/>, combined with fresh packed noise via flow-matching <c>AddNoise</c>: <c>noisy = (1-sigma) * source + sigma * noise</c>.
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint path), the packed source latent is returned as the second tuple element instead of being disposed — the caller reuses it per step for blending. Returns null source for txt2img and for img2img without a mask.</para></summary>
    private (Tensor packedLatent, Tensor? packedSourceLatent) BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape latentShape,
        TensorShape packedShape,
        int latentH, int latentW, int seed, int startStep,
        bool keepSourceLatent)
    {
        Tensor unpackedNoise = TakeOrCreateNoise(request, latentShape, seed);
        Tensor packedNoise = PackLatent(unpackedNoise, latentH, latentW);
        unpackedNoise.Dispose();

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceUnpacked = _vaeEncoder!.Encode(VaeBackend, img2img.SourceImage);  // LOAD-BEARING for VaeDevice: PackLatent below is a host loop
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor sourcePacked = PackLatent(sourceUnpacked, latentH, latentW);
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

        // T2I path: scale packed noise by initSigma.
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

    /// <summary>Sums two Flux ControlNet residual stacks element-wise (multi-ControlNet stacking). Consumes both inputs' tensors and returns a stack of fresh sums.</summary>
    private Adapters.FluxControlNetResiduals SumControlNetResiduals(
        Adapters.FluxControlNetResiduals acc, Adapters.FluxControlNetResiduals next)
    {
        if (acc.DoubleBlockResiduals.Length != next.DoubleBlockResiduals.Length
            || acc.SingleBlockResiduals.Length != next.SingleBlockResiduals.Length)
        {
            acc.DisposeAll();
            next.DisposeAll();
            throw new InvalidOperationException(
                "Stacked Flux ControlNets produced mismatched residual counts " +
                $"({acc.DoubleBlockResiduals.Length}+{acc.SingleBlockResiduals.Length} vs " +
                $"{next.DoubleBlockResiduals.Length}+{next.SingleBlockResiduals.Length}) — all stacked adapters " +
                "must share the same block depths.");
        }
        SumInto(acc.DoubleBlockResiduals, next.DoubleBlockResiduals);
        SumInto(acc.SingleBlockResiduals, next.SingleBlockResiduals);
        return acc;
    }

    private void SumInto(Tensor[] acc, Tensor[] next)
    {
        for (int i = 0; i < acc.Length; i++)
        {
            Tensor sum = new Tensor(acc[i].Shape, acc[i].DType);
            Backend.Add(sum, acc[i], next[i]);
            acc[i].Dispose();
            next[i].Dispose();
            acc[i] = sum;
        }
    }

    /// <summary>Builds a prompt-cache key from the CLIP-L token ids, the CLIP EOS position (the pooled vector depends on it), and the T5 token ids, in one flat array (lengths make the encoding unambiguous).</summary>
    private static int[] BuildPromptCacheKey(int[] tokenIdsL, int eosPositionL, int[] tokenIdsT5)
    {
        int[] key = new int[tokenIdsL.Length + 2 + tokenIdsT5.Length];
        tokenIdsL.CopyTo(key, 0);
        key[tokenIdsL.Length] = eosPositionL;
        key[tokenIdsL.Length + 1] = tokenIdsL.Length;
        tokenIdsT5.CopyTo(key, tokenIdsL.Length + 2);
        return key;
    }


    private static void LogTensorStats(string name, Tensor tensor)
    {
        if (!StatsEnabled) return;
        ReadOnlySpan<float> data = tensor.AsReadOnlySpan<float>();
        float min = float.MaxValue, max = float.MinValue, sum = 0;
        int nanCount = 0, infCount = 0, zeroCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v == 0f) zeroCount++;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        float mean = data.Length > 0 ? sum / data.Length : 0;
        Logs.Debug($"  [{name}] shape={tensor.Shape} dtype={tensor.DType} min={min:E3} max={max:E3} mean={mean:E3} nan={nanCount} inf={infCount} zero={zeroCount}/{data.Length}");
    }

    /// <summary>Logs per-latent-channel mean for a packed [B, S, 64] tensor (velocity or latent). Each latent channel occupies 4 contiguous slots in the feature dim (c*4 .. c*4+3 for 2x2 patch).</summary>
    private static void LogPerLatentChannelMeanPacked(string name, Tensor packed)
    {
        if (!StatsEnabled) return;
        long batch = packed.Shape[0];
        long seqLen = packed.Shape[1];
        long featDim = packed.Shape[2];
        int latentChannels = (int)(featDim / 4);
        float* ptr = (float*)packed.DataPointer;
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.Append($"  [{name}] per-lat-ch mean: ");
        for (int c = 0; c < latentChannels; c++)
        {
            double sum = 0;
            long count = 0;
            for (long b = 0; b < batch; b++)
            {
                for (long s = 0; s < seqLen; s++)
                {
                    long baseIdx = (b * seqLen + s) * featDim + c * 4;
                    sum += ptr[baseIdx + 0];
                    sum += ptr[baseIdx + 1];
                    sum += ptr[baseIdx + 2];
                    sum += ptr[baseIdx + 3];
                    count += 4;
                }
            }
            float mean = (float)(sum / count);
            sb.Append($"c{c}={mean:+0.000;-0.000} ");
        }
        Logs.Debug(sb.ToString());
    }

    /// <summary>Logs per-channel statistics for a 4D NCHW tensor. Useful for diagnosing color channel imbalances.</summary>
    private static void LogPerChannelStats(string name, Tensor tensor)
    {
        if (!StatsEnabled) return;
        int channels = (int)tensor.Shape[1];
        int spatial = (int)(tensor.Shape[2] * tensor.Shape[3]);
        float* ptr = (float*)tensor.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float cMin = float.MaxValue, cMax = float.MinValue, cSum = 0;
            for (int i = 0; i < spatial; i++)
            {
                float v = ptr[c * spatial + i];
                if (v < cMin) cMin = v;
                if (v > cMax) cMax = v;
                cSum += v;
            }
            float cMean = spatial > 0 ? cSum / spatial : 0;
            Logs.Info($"  [{name}] ch{c}: min={cMin:F4} max={cMax:F4} mean={cMean:F4}");
        }
    }

    /// <summary>Concatenates two packed-form tensors <c>[1, S, D1]</c> and <c>[1, S, D2]</c> along the feature dim → <c>[1, S, D1+D2]</c>. Used by FLUX.1 Tools to glue the packed control latent onto the packed noise before the wider <c>x_embedder</c>. Both inputs must share the same batch and sequence length; F32 only.</summary>
    private static Tensor ConcatPackedFeatureDim(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 3 || b.Shape.Rank != 3 || a.Shape[0] != b.Shape[0] || a.Shape[1] != b.Shape[1])
        {
            throw new ArgumentException($"ConcatPackedFeatureDim requires [B, S, D] tensors with matching B and S; got {a.Shape} and {b.Shape}.");
        }
        long batch = a.Shape[0];
        long seqLen = a.Shape[1];
        long dimA = a.Shape[2];
        long dimB = b.Shape[2];
        Tensor output = new Tensor(new TensorShape(batch, seqLen, dimA + dimB), DType.F32);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float* op = (float*)output.DataPointer;
        long aBytes = dimA * sizeof(float);
        long bBytes = dimB * sizeof(float);
        for (long bi = 0; bi < batch; bi++)
        {
            for (long s = 0; s < seqLen; s++)
            {
                long aOff = (bi * seqLen + s) * dimA;
                long bOff = (bi * seqLen + s) * dimB;
                long oOff = (bi * seqLen + s) * (dimA + dimB);
                Buffer.MemoryCopy(ap + aOff, op + oOff, aBytes, aBytes);
                Buffer.MemoryCopy(bp + bOff, op + oOff + dimA, bBytes, bBytes);
            }
        }
        return output;
    }

    /// <summary>Concatenates two packed-form tensors <c>[1, Sa, D]</c> and <c>[1, Sb, D]</c> along the SEQUENCE dim → <c>[1, Sa+Sb, D]</c>. Used by Flux Kontext to append the packed reference-image tokens after the packed noise tokens before the transformer. Both inputs must share batch and feature dim; F32 only.</summary>
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

    /// <summary>Zeroes the masked region of a [-1,1] RGB source — <c>output = source · (1 − round(mask))</c>, mask broadcast over the 3 channels — producing FLUX.1 Fill's masked-image input (0 = the VAE's mid-gray). The mask is BINARIZED here (Comfy InpaintModelConditioning rounds it for the pixels); a soft/blurred mask would otherwise paint a half-darkened ring into the conditioning image that the model reproduces at the seam. The 256-channel conditioning mask stays continuous.</summary>
    internal static Tensor MaskPixelsToNeutral(Tensor source, Tensor mask)
    {
        long h = source.Shape[2];
        long w = source.Shape[3];
        Tensor output = new Tensor(source.Shape, DType.F32);
        float* sp = (float*)source.DataPointer;
        float* mp = (float*)mask.DataPointer;
        float* op = (float*)output.DataPointer;
        long plane = h * w;
        for (int c = 0; c < 3; c++)
        {
            long cOff = c * plane;
            for (long i = 0; i < plane; i++)
            {
                op[cOff + i] = mp[i] >= 0.5f ? 0.0f : sp[cOff + i];
            }
        }
        return output;
    }

    /// <summary>Expands a pixel mask <c>[1, 1, H·8, W·8]</c> to FLUX.1 Fill's 64-channel latent-resolution form <c>[1, 64, H, W]</c>: channel <c>sy·8+sx</c> holds the mask pixel at sub-position (sy, sx) of each 8×8 block (diffusers FluxFillPipeline's view→permute→reshape).</summary>
    internal static Tensor FillMaskToLatentChannels(Tensor maskPixel, int latentH, int latentW)
    {
        Tensor output = new Tensor(new TensorShape(1, 64, latentH, latentW), DType.F32);
        float* mp = (float*)maskPixel.DataPointer;
        float* op = (float*)output.DataPointer;
        int pixW = latentW * 8;
        for (int sy = 0; sy < 8; sy++)
        {
            for (int sx = 0; sx < 8; sx++)
            {
                long cOff = (long)(sy * 8 + sx) * latentH * latentW;
                for (int i = 0; i < latentH; i++)
                {
                    long rowOff = (long)(i * 8 + sy) * pixW;
                    for (int j = 0; j < latentW; j++)
                    {
                        op[cOff + (long)i * latentW + j] = mp[rowOff + j * 8 + sx];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Packs a latent tensor from [B, C, H, W] to [B, H/2*W/2, C*4]. Rearranges 2x2 spatial patches into channel dimension.</summary>
    internal static Tensor PackLatent(Tensor latent, int h, int w)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4; // 16 * 4 = 64
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
                    int outBase = (b * seqLen + seqIdx) * patchDim;

                    // For each channel, gather 2x2 patch
                    for (int c = 0; c < channels; c++)
                    {
                        int inChannelBase = (b * channels + c) * h * w;
                        int patchBase = outBase + c * 4;

                        outPtr[patchBase + 0] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 1] = inPtr[inChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)];
                        outPtr[patchBase + 2] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)];
                        outPtr[patchBase + 3] = inPtr[inChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)];
                    }
                }
            }
        }

        return packed;
    }

    /// <summary>Unpacks a latent tensor from [B, H/2*W/2, C*4] back to [B, C, H, W]. Does not dispose the input. Internal so <see cref="ChromaPipeline"/> shares the identical packed layout.</summary>
    internal static Tensor UnpackLatent(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = 16; // Flux always uses 16 latent channels
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4;
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
                    int inBase = (b * seqLen + seqIdx) * patchDim;

                    for (int c = 0; c < channels; c++)
                    {
                        int outChannelBase = (b * channels + c) * h * w;
                        int patchBase = inBase + c * 4;

                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 0)] = inPtr[patchBase + 0];
                        outPtr[outChannelBase + (ph * 2 + 0) * w + (pw * 2 + 1)] = inPtr[patchBase + 1];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 0)] = inPtr[patchBase + 2];
                        outPtr[outChannelBase + (ph * 2 + 1) * w + (pw * 2 + 1)] = inPtr[patchBase + 3];
                    }
                }
            }
        }

        return unpacked;
    }

    /// <summary>Performs a scheduler step on packed latent tokens. Euler: x_next = x + v * dt, where dt = sigma_next - sigma.</summary>
    private static void SchedulerStepPacked(Tensor output, Tensor velocity, Tensor sample,
        FlowMatchEulerDiscreteScheduler scheduler, int stepIndex)
    {
        // The scheduler operates on flat data regardless of shape
        scheduler.Step(output, velocity, sample, stepIndex);
    }

    /// <summary>Extracts the hidden state at the EOS token position from [B, seqLen, hiddenSize] to [B, hiddenSize]. Used to get CLIP-L pooled output for Flux (no text_projection needed).</summary>
    private static Tensor ExtractEosHiddenState(Tensor hidden, int eosPosition)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int hiddenSize = (int)hidden.Shape[2];

        TensorShape pooledShape = new TensorShape(batch, hiddenSize);
        Tensor pooled = new Tensor(pooledShape, DType.F32);

        float* srcPtr = (float*)hidden.DataPointer;
        float* dstPtr = (float*)pooled.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int srcOffset = (b * seqLen + eosPosition) * hiddenSize;
            int dstOffset = b * hiddenSize;
            for (int d = 0; d < hiddenSize; d++)
            {
                dstPtr[dstOffset + d] = srcPtr[srcOffset + d];
            }
        }

        return pooled;
    }
}
