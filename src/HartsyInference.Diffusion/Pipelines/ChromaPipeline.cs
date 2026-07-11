using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Chroma text-to-image pipeline (<c>lodestones/Chroma</c>). T5-XXL → ChromaTransformer denoising
/// with flow matching (dynamic shift, Flux-style) → 16-channel Flux VAE decode → RGB.
///
/// Differences from Flux that the pipeline must handle (see <see cref="ChromaTransformer"/> and
/// <c>diffusers/pipelines/chroma/pipeline_chroma.py</c>):
/// <list type="bullet">
///   <item><b>T5-only encode</b> — no CLIP. Caller passes already-tokenized T5 IDs and an attention mask.</item>
///   <item><b>"First padding token unmasked"</b> — Chroma propagates a per-token attention mask through every
///         transformer block. The mask is built from the tokenizer mask via
///         <c>(arange(seq_len) &lt;= text_lens)</c> i.e. all real tokens plus exactly one extra unmasked
///         padding slot at the EOS position.</item>
///   <item><b>True CFG</b> — dual transformer pass. Default cfg=5.0, steps=35.</item>
///   <item><b>Dynamic flow-match shift</b> via <see cref="FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift"/>
///         (Flux-style; same base/max constants).</item>
///   <item><b>Latent packing</b> identical to Flux (2x2 patchify into channel dim, 16ch → 64dim).</item>
/// </list>
/// VRAM eviction at the transformer→VAE boundary mirrors <see cref="Sd3Pipeline"/> / <see cref="FluxPipeline"/>.
/// Img2img / inpaint follows the Flux pattern (packed AddNoise at sigma[startStep], per-step packed mask blend,
/// final pixel recomposite) — pass an <see cref="ImageToImageRequest"/> and construct with a <see cref="VaeEncoder"/>.</summary>
public sealed unsafe class ChromaPipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly ChromaTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly ChromaConfig _config;
    private readonly float _schedulerShiftFallback;

    /// <summary>Keeps the DiT weights GPU-resident across generations (skips the post-loop
    /// FreeWeights + next-gen re-upload). The T5-XXL cannot coexist with the resident DiT on smaller cards, so a
    /// prompt-cache MISS under this flag frees the DiT first, encodes, then re-preloads — repeat prompts skip both.
    /// Standard-profile default ON (HARTSY_KEEP_MODELS=0 disables) — the miss-path eviction above is what keeps smaller cards viable even with residency on.</summary>
    private static readonly bool KeepModelsResident =
        EnvSwitch.IsEnabled("HARTSY_KEEP_MODELS", defaultOn: true);
    private bool _ditResident;

    // Prompt-embedding cache (one cond + one uncond, last-used), keyed on the T5 token ids — the Krea2 pattern.
    // A hit skips the whole T5 phase (preload + encode + free). The derived transformer-side masks are cached
    // alongside (they are pure functions of the tokenizer masks).
    private int[]? _cachedCondKey;
    private Tensor? _cachedCond;
    private Tensor? _cachedCondMask;
    private int[]? _cachedUncondKey;
    private Tensor? _cachedUncond;
    private Tensor? _cachedUncondMask;

    /// <summary>Creates a new Chroma pipeline with all components pre-loaded. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">T5-XXL text encoder (joint_attention_dim = 4096, max length 512).</param>
    /// <param name="transformer">Chroma transformer (loaded with <see cref="ChromaConfig"/>).</param>
    /// <param name="vaeDecoder">16-channel Flux VAE decoder (use <see cref="VaeConfig.Flux"/>).</param>
    /// <param name="config">Chroma configuration (use <see cref="ChromaConfig.V1"/> for the v1 release).</param>
    /// <param name="schedulerShift">Fallback static shift if the runtime ever needs one. Chroma uses a dynamic
    /// shift derived from the image sequence length per call; this constant is only used if the call site
    /// somehow short-circuits the dynamic path.</param>
    public ChromaPipeline(IBackend backend, T5TextEncoder t5, ChromaTransformer transformer,
        VaeDecoder vaeDecoder, ChromaConfig config, float schedulerShift = 3.0f)
        : this(backend, t5, transformer, vaeDecoder, vaeEncoder: null, config, schedulerShift)
    {
    }

    /// <summary>Creates a new Chroma pipeline with both VAE halves loaded. Required for img2img / inpaint (pass an <see cref="ImageToImageRequest"/> to <see cref="GenerateFromTokens"/>).</summary>
    public ChromaPipeline(IBackend backend, T5TextEncoder t5, ChromaTransformer transformer,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, ChromaConfig config, float schedulerShift = 3.0f)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _config = config;
        _schedulerShiftFallback = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized T5 input plus attention masks. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>: an <see cref="ImageToImageRequest"/> encodes the source through the 16-channel Flux VAE, packs it (2×2 patchify), and injects flow-matching noise at <c>sigma[startStep]</c> — requires a <see cref="VaeEncoder"/> on construction. A <c>Mask</c> additionally enables blend-on-vanilla inpaint (per-step packed blend + final pixel recomposite, same as <see cref="FluxPipeline"/>). Strength=0 short-circuits to byte-identical pass-through.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from the T5 tokenizer.</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs (same length as <paramref name="promptTokenIdsT5"/>).</param>
    /// <param name="promptAttentionMaskT5">Tokenizer attention mask for the prompt (1=real token, 0=pad). Required
    /// — Chroma needs this to compute the "first padding token unmasked" extension.</param>
    /// <param name="negativeAttentionMaskT5">Tokenizer attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[] promptAttentionMaskT5,
        int[] negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (promptAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(promptAttentionMaskT5),
                "Chroma requires the tokenizer attention mask (1=real, 0=pad) so the pipeline can " +
                "compute the 'first padding token unmasked' rule.");
        if (negativeAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(negativeAttentionMaskT5),
                "Chroma requires the tokenizer attention mask for the negative prompt as well.");

        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Chroma.Width;
        int height = request.Height ?? GenerationDefaults.Chroma.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;
        bool useCfg = cfgScale > 1.0f;

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
        Logs.Info($"Chroma {opMode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts with T5-XXL ─────────────────────────────────
        // Prompt-embedding cache: identical token ids reuse the previous gen's hidden states + derived masks —
        // the whole T5 phase (preload + encode + free) vanishes for repeat prompts (seed-only changes).
        bool condHit = _cachedCond is not null
            && _cachedCondKey is not null && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIdsT5);
        bool uncondHit = !useCfg || (_cachedUncond is not null
            && _cachedUncondKey is not null && _cachedUncondKey.AsSpan().SequenceEqual(negativePromptTokenIdsT5));
        Tensor condContext;
        Tensor? condMask;
        Tensor? uncondContext = null;
        Tensor? uncondMask = null;
        if (condHit && uncondHit)
        {
            condContext = _cachedCond!;
            condMask = _cachedCondMask;
            if (useCfg)
            {
                uncondContext = _cachedUncond;
                uncondMask = _cachedUncondMask;
            }
            Logs.Info("[Chroma] prompt-embedding cache hit — T5 phase skipped");
        }
        else
        {
            Logs.Info("Encoding text with T5-XXL...");
            if (_ditResident)
            {
                // The T5 cannot coexist with the resident DiT (HARTSY_KEEP_MODELS); evict for this
                // new-prompt generation and re-preload below. The cross-generation step graph bakes the
                // WEIGHT device pointers — invalidate before the free or a later replay is a CUDA 700.
                _transformer.InvalidateStepGraph(Backend);
                Backend.Sync();
                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }
            // Bulk-upload T5 weights once (~5 GB on T5-XXL) so the encoder's many kernels don't
            // each pay a per-op cache-miss H2D transfer. Paired with FreeWeights below the VAE
            // handoff. No-op on backends without a weight cache.
            Backend.PreloadWeights(_t5.EnumerateWeights());

            int[][] batchT5 = [promptTokenIdsT5];
            int[][] batchMask = [promptAttentionMaskT5];
            condContext = _t5.Encode(Backend, batchT5, batchMask);

            if (useCfg)
            {
                int[][] negBatchT5 = [negativePromptTokenIdsT5];
                int[][] negBatchMask = [negativeAttentionMaskT5];
                uncondContext = _t5.Encode(Backend, negBatchT5, negBatchMask);
            }

            // Trim the padded context to Chroma's kept tokens instead of masking. Chroma's transformer-side
            // rule (pipeline_chroma.py:249-252) keeps positions i <= text_len — all real tokens plus exactly
            // one unmasked padding slot — and masks the rest OUT of every attention. Masked-out tokens
            // contribute nothing to any kept token (and only kept image tokens reach the output), so slicing
            // them off is EXACT while shrinking every joint-sequence GEMM and making all 57 SDPAs mask-free
            // (~500 dead tokens of a 512-pad prompt at 1024²: ~10% of the joint sequence, ~20% of the
            // attention score work, and the 85 MB additive mask disappears). A prompt that fills the full
            // window trims nothing and still runs unmasked — all positions are kept either way.
            condContext = TrimContextToKeptTokens(condContext, promptAttentionMaskT5);
            if (uncondContext is not null)
                uncondContext = TrimContextToKeptTokens(uncondContext, negativeAttentionMaskT5);
            condMask = null;
            uncondMask = null;

            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms " +
                $"(context trimmed to {condContext.Shape[1]}/{uncondContext?.Shape[1].ToString() ?? "-"} kept tokens)");

            // The conditioning is host-materialized (the trim reads it) so it survives activation reclaims;
            // cache it. Eviction FreeWeights first: the step-graph warm-up pins the cached contexts
            // device-resident (no-op when they were never pinned — the Z-Image cached-caption pattern).
            _ = condContext.DataPointer;
            if (uncondContext is not null) _ = uncondContext.DataPointer;
            if (_cachedCond is not null)
            {
                Backend.FreeWeights(new[] { _cachedCond });
                _cachedCond.Dispose();
            }
            _cachedCondMask?.Dispose();
            _cachedCond = condContext;
            _cachedCondMask = condMask;
            _cachedCondKey = (int[])promptTokenIdsT5.Clone();
            if (useCfg)
            {
                if (_cachedUncond is not null)
                {
                    Backend.FreeWeights(new[] { _cachedUncond });
                    _cachedUncond.Dispose();
                }
                _cachedUncondMask?.Dispose();
                _cachedUncond = uncondContext;
                _cachedUncondMask = uncondMask;
                _cachedUncondKey = (int[])negativePromptTokenIdsT5.Clone();
            }
        }

        // ── 2. Set up dynamic flow-match scheduler ────────────────────────
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);

        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        // Touch the fallback shift field so the field isn't unused on builds that don't reach the dynamic path.
        _ = _schedulerShiftFallback;
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial packed latent ────────────────────────────────
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

        // ── 4. Denoising loop ────────────────────────────────────────────
        // Free T5 weights from VRAM before uploading the transformer. T5-XXL is ~5 GB; Chroma is
        // ~9 GB FP8. Without this free, both sit in VRAM together (14 GB) and OOM on a 12 GB card.
        Backend.FreeWeights(_t5.EnumerateWeights());
        Backend.Sync();

        // Bulk-upload transformer weights before the denoise loop. Chroma is Flux-derived
        // (~12 GB at FP16, ~9 GB at FP8) — without preload the first step would pay cache-miss
        // overhead for every block. Paired with FreeWeights below the VAE handoff.
        Backend.PreloadWeights(_transformer.EnumerateWeights());
        _ditResident = true;

        Logs.Info("Starting Chroma denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int txtSeqLen = (int)condContext.Shape[1];

        // Drain-free fast path (plain t2i / img2img): the latent stays device-resident across the whole loop
        // and CFG combine + Euler run as ONE in-place device op (CfgEulerStep: uncond-anchored CFG then
        // z += v·dt; gw=1 degenerates to the plain Euler step). Masked inpaint keeps the host branch — it
        // must read/rebuild the latent on the host every step. The CFG pair runs through ForwardPaired
        // (one modulation-table build per step, and — default-on — the per-generation step CUDA graph).
        bool drainFree = !isMaskedInpaint;

        // Step-graph mode: route the latent through the transformer's FIXED buffer so the captured graph's
        // baked address stays valid across steps (CfgEulerStep updates it in place at the same address).
        // The fixed tensor is transformer-owned: never disposed here, never DataPointer-read directly —
        // previews and the final unpack go through SnapshotGraphLatent.
        bool graphMode = drainFree
            && Models.Denoisers.DiTBlocks.DitStepGraph.EnabledDefaultOn && Backend.StepGraphSupported;
        if (graphMode)
        {
            Tensor routed = _transformer.PrepareGraphLatent(Backend, packedLatent);
            packedLatent.Dispose();
            packedLatent = routed;
        }

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            if (drainFree)
            {
                (Tensor condNoise, Tensor? uncondNoise, bool callerOwns) = _transformer.ForwardPaired(
                    Backend, packedLatent, condContext, useCfg ? uncondContext : null, sigma,
                    hPacked, wPacked, condMask, useCfg ? uncondMask : null);
                if (useCfg)
                {
                    Backend.CfgEulerStep(packedLatent, condNoise, uncondNoise!, cfgScale, scheduler.Dt(i));
                }
                else
                {
                    Backend.CfgEulerStep(packedLatent, condNoise, condNoise, 1.0f, scheduler.Dt(i));
                }
                if (callerOwns)
                {
                    condNoise.Dispose();
                    uncondNoise?.Dispose();
                }
            }
            else
            {
            Tensor noisePred;
            if (useCfg)
            {
                noisePred = ClassifierFreeGuidanceStep(packedLatent, sigma,
                    condContext, condMask,
                    uncondContext!, uncondMask,
                    txtSeqLen, hPacked, wPacked, cfgScale);
            }
            else
            {
                noisePred = _transformer.Forward(Backend, packedLatent, condContext, sigma,
                    txtSeqLen, hPacked, wPacked, condMask);
            }

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
                    Tensor freshPackedNoise = PackLatent(SeedGenerator.CreateNoise(latentShape, seed + nextStep), latentH, latentW);
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
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            // Same packed layout as Flux.1 (Chroma reuses the Flux VAE). Unpack inline so preview renders.
            // On the drain-free path the unpack reads the device-resident latent (a D2H sync per frame), so
            // previews emit every 4th step + final instead of every step; the host path keeps every step.
            // Drain-free previews: every 4th step, but NOT the final step — the finished image follows
            // immediately, and the final-step latent preview would add a redundant full drain + D2H +
            // convert right before the end-of-loop snapshot does the same work.
            bool emitPreview = onProgress is not null
                && (!drainFree || ((i - startStep) % 4 == 3 && i != steps - 1));
            if (emitPreview)
            {
                // Graph mode: never read the fixed latent's DataPointer (D2H would evict the buffer the
                // captured graph points at) — preview from a device-copied snapshot instead.
                Tensor previewSource = graphMode ? _transformer.SnapshotGraphLatent(Backend) : packedLatent;
                Tensor previewLatent = LatentPreview.UnpackFluxStylePacked(previewSource, latentH, latentW, channels: 16);
                if (graphMode) previewSource.Dispose();
                try
                {
                    onProgress!.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                    {
                        Latent = previewLatent,
                        LatentArch = LatentArchitecture.Chroma,
                    });
                }
                finally { previewLatent.Dispose(); }
            }
            else if (onProgress is not null)
            {
                onProgress.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
            }
        }

        // condContext/condMask/uncond* are cross-generation caches — not disposed here.
        packedSourceLatent?.Dispose();
        packedMask?.Dispose();

        // Graph mode: swap the transformer-owned fixed latent for a disposable device snapshot. The captured
        // graph itself PERSISTS across generations — this pipeline never calls FreeActivations, so the fixed
        // boundary buffers stay live and a repeat-prompt generation replays every step with zero re-warm /
        // re-capture cost. Anything that frees the baked pointers invalidates first: the two DiT
        // FreeWeights sites above, and PrepareGraphLatent on a resolution change.
        if (graphMode)
        {
            packedLatent = _transformer.SnapshotGraphLatent(Backend);
        }

        ChromaTransformer.DumpFinalLatent(packedLatent);

        // Free transformer + T5 weights from GPU before VAE decode (mirrors SD3/Flux/AuraFlow pattern).
        // Under HARTSY_KEEP_MODELS the DiT stays resident (~9 GB fp8; the full-res VAE decode's banded im2col
        // fits beside it, falling back to tiles if not).
        Backend.Sync();
        if (!KeepModelsResident)
        {
            _transformer.InvalidateStepGraph(Backend);
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        Backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. Unpack latent: [1, seqLen, 64] → [1, 16, latentH, latentW] ──
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();

        // ── 6. VAE decode ────────────────────────────────────────────────
        Backend.PreloadWeights(_vaeDecoder.EnumerateWeights());
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile. Internal fast-path
        // skips tiling when the latent fits in a single tile, so small images pay no overhead.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, unpackedLatent);
        unpackedLatent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 7. Pixel-space recomposite for masked inpaint: paste decoded over source where mask=1.
        //    Suppresses VAE encode/decode drift in unmasked regions (same as SDXL / Flux).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Chroma {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the initial packed latent for Chroma denoising. T2I: fresh (or caller-injected) noise packed and scaled by the scheduler's initial sigma. Img2img: VAE-encoded source (16 channels) packed via <see cref="PackLatent"/>, combined with fresh packed noise via flow-matching <c>AddNoise</c>: <c>noisy = (1-sigma) * source + sigma * noise</c>.
    /// <para>When <paramref name="keepSourceLatent"/> is true (masked inpaint), the packed source latent is returned as the second tuple element for per-step blending. Caller disposes both. Source is null for txt2img and plain img2img.</para></summary>
    private (Tensor packedLatent, Tensor? packedSourceLatent) BuildInitialPackedLatent(
        TextToImageRequest request,
        FlowMatchEulerDiscreteScheduler scheduler,
        TensorShape latentShape,
        TensorShape packedShape,
        int latentH, int latentW, int seed, int startStep,
        bool keepSourceLatent)
    {
        // Honor a caller-supplied initial noise tensor (fixed-seed reproduction) when present; otherwise
        // sample fresh noise from the seed. The tensor is in unpacked latent layout [1, 16, latentH, latentW];
        // PackLatent takes ownership and disposes it.
        Tensor initialNoise = request.InitialNoise ?? SeedGenerator.CreateNoise(latentShape, seed);
        if (!initialNoise.Shape.Equals(latentShape))
            throw new ArgumentException($"InitialNoise shape {initialNoise.Shape} != expected {latentShape}.");
        Tensor packedNoise = PackLatent(initialNoise, latentH, latentW);

        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceUnpacked = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor sourcePacked = PackLatent(sourceUnpacked, latentH, latentW);
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

    /// <summary>Runs classifier-free guidance with Chroma's cond-anchored convention: <c>noise_pred = cond + cfg_scale * (cond - uncond)</c>. Chroma does real dual-pass CFG against a negative prompt; only the combine baseline differs from the standard formula.</summary>
    private Tensor ClassifierFreeGuidanceStep(
        Tensor packedLatent, float sigma,
        Tensor condContext, Tensor? condMask,
        Tensor uncondContext, Tensor? uncondMask,
        int txtSeqLen, int hPacked, int wPacked, float cfgScale)
    {
        // Note: the negative prompt may have a different sequence length from the positive prompt; the
        // transformer is fully shape-polymorphic on txtSeqLen so each pass uses its own value.
        int condTxtLen = (int)condContext.Shape[1];
        int uncondTxtLen = (int)uncondContext.Shape[1];

        Tensor uncondNoise = _transformer.Forward(Backend, packedLatent, uncondContext, sigma,
            uncondTxtLen, hPacked, wPacked, uncondMask);
        Tensor condNoise = _transformer.Forward(Backend, packedLatent, condContext, sigma,
            condTxtLen, hPacked, wPacked, condMask);

        // Touch txtSeqLen so the field isn't flagged unused (it remains for callers that need to introspect).
        _ = txtSeqLen;

        // Chroma uses STANDARD uncond-anchored CFG: uncond + scale*(cond - uncond). Both diffusers
        // pipeline_chroma.py and ComfyUI use this; the prior cond-anchored form over-guided by one (cond-uncond)
        // unit (cfg behaved like cfg+1 → over-saturation). Verified against the reference implementations.
        Tensor output = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
        uncondNoise.Dispose();
        condNoise.Dispose();
        return output;
    }

    /// <summary>Slices the encoded [1, seqLen, hidden] T5 context down to Chroma's kept tokens
    /// (<c>text_len + 1</c>: all real tokens plus the one unmasked padding slot). Exact — the dropped rows are
    /// masked out of every attention by the transformer-side rule, so they can never influence the output.
    /// Consumes (disposes) the input when a trim happens; the read also host-materializes the result.
    /// Internal so <see cref="ChromaRadiancePipeline"/> shares the identical text path.</summary>
    internal static Tensor TrimContextToKeptTokens(Tensor context, int[] tokenizerMask)
    {
        int seqLen = (int)context.Shape[1];
        int hidden = (int)context.Shape[2];
        int textLen = 0;
        for (int i = 0; i < tokenizerMask.Length; i++) if (tokenizerMask[i] != 0) textLen++;
        int kept = Math.Min(textLen + 1, seqLen);
        if (kept == seqLen)
        {
            _ = context.DataPointer;
            return context;
        }
        Tensor trimmed = new Tensor(new TensorShape(1, kept, hidden), DType.F32);
        long bytes = (long)kept * hidden * sizeof(float);
        Buffer.MemoryCopy((void*)context.DataPointer, (void*)trimmed.DataPointer, bytes, bytes);
        context.Dispose();
        return trimmed;
    }

    /// <summary>Builds Chroma's transformer-side text mask. Per pipeline_chroma.py:249-252, the mask used inside
    /// the transformer is <c>(arange(seq_len) &lt;= text_len)</c>: every real token plus exactly one extra
    /// padding slot kept unmasked. <paramref name="tokenizerMask"/> is the standard tokenizer mask
    /// (1=real, 0=pad). Returns a [batch, seq_len] F32 tensor with the extension applied.
    /// Internal so <see cref="ChromaRadiancePipeline"/> shares the identical text path.</summary>
    internal static Tensor BuildChromaTextMask(int[] tokenizerMask, int batchSize)
    {
        int seqLen = tokenizerMask.Length;
        TensorShape shape = new TensorShape(batchSize, seqLen);
        Tensor mask = new Tensor(shape, DType.F32);

        // text_len = number of real tokens (sum of the tokenizer mask). The "<=" rule keeps positions
        // [0, text_len], which is text_len + 1 unmasked entries — text_len real plus one padding slot.
        int textLen = 0;
        for (int i = 0; i < seqLen; i++) if (tokenizerMask[i] != 0) textLen++;

        float* ptr = (float*)mask.DataPointer;
        for (int b = 0; b < batchSize; b++)
        {
            int baseOffset = b * seqLen;
            for (int i = 0; i < seqLen; i++)
                ptr[baseOffset + i] = i <= textLen ? 1.0f : 0.0f;
        }
        return mask;
    }

    /// <summary>Packs a latent tensor [B, 16, H, W] → [B, H/2 * W/2, 64] (2x2 patchify, identical to Flux).</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int hPacked = h / 2;
        int wPacked = w / 2;
        int patchDim = channels * 4;
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

        latent.Dispose();
        return packed;
    }

    /// <summary>Unpacks [B, S, 64] → [B, 16, H, W]. Inverse of <see cref="PackLatent"/>.</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0];
        int channels = 16;
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
}
