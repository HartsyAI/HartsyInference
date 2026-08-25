using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Chroma Radiance text-to-image pipeline (<c>lodestones/Chroma1-Radiance</c>) — pixel-space, VAE-free. T5-XXL encode (identical text path to <see cref="ChromaPipeline"/>, including the "first padding token unmasked" mask rule) → <see cref="ChromaRadianceTransformer"/> denoising directly on RGB in [-1, 1] → bytes. Sampling differences vs classic Chroma:
/// <list type="bullet">
///   <item><b>x0 prediction</b> — the model predicts the clean image. Each step converts to velocity via <see cref="X0Prediction.ToVelocity"/> (<c>v = (x_t − x0) / max(t, ε)</c>, matching ComfyUI), CFG-combines on v, then takes the flow-match Euler step.</item>
///   <item><b>Static shift 1.0</b> (ComfyUI's Chroma sampling default — no dynamic shift), default 50 steps, CFG 3.5.</item>
///   <item><b>Pixel space</b> — no latent packing, no VAE. Dimensions are padded up to a multiple of the 16-px patch and the output is cropped back. Previews are the in-flight image itself (<see cref="LatentArchitecture.ChromaRadiance"/>).</item>
///   <item><b>Img2img / inpaint without a VAE</b> — the source image IS the clean sample: it's padded to the patch grid and noised directly via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Masked inpaint blends per step in pixel space (mask used at full resolution — no downsample).</item>
/// </list></summary>
public sealed unsafe class ChromaRadiancePipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly ChromaRadianceTransformer _transformer;
    private readonly ChromaRadianceConfig _config;

    // Prompt-embedding cache + DiT residency (the ChromaPipeline pattern): repeat prompts skip the whole T5
    // phase AND the ~9 GB transformer re-upload — the DiT stays device-resident across generations and is only
    // evicted when a new prompt needs the T5 on the card.
    private int[]? _cachedCondKey;
    private Tensor? _cachedCond;
    private int[]? _cachedUncondKey;
    private Tensor? _cachedUncond;
    private bool _ditResident;

    /// <summary>Creates a new Chroma Radiance pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">T5-XXL text encoder (joint_attention_dim = 4096, max length 512).</param>
    /// <param name="transformer">Radiance transformer (loaded with <see cref="ChromaRadianceConfig"/>).</param>
    /// <param name="config">Radiance configuration (use <see cref="ChromaRadianceConfig.V1"/> or <c>FromWeights</c>).</param>
    public ChromaRadiancePipeline(IBackend backend, T5TextEncoder t5, ChromaRadianceTransformer transformer,
        ChromaRadianceConfig config)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized T5 input plus attention masks. API mirrors <see cref="ChromaPipeline.GenerateFromTokens"/>.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from the T5 tokenizer.</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs.</param>
    /// <param name="promptAttentionMaskT5">Tokenizer attention mask for the prompt (1=real token, 0=pad).</param>
    /// <param name="negativeAttentionMaskT5">Tokenizer attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint (pixel-space — no VAE encoder needed; strength=0 short-circuits to byte-identical pass-through).</param>
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
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);

        if (promptAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(promptAttentionMaskT5),
                "Chroma Radiance requires the tokenizer attention mask (1=real, 0=pad) so the pipeline can " +
                "compute the 'first padding token unmasked' rule.");
        if (negativeAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(negativeAttentionMaskT5),
                "Chroma Radiance requires the tokenizer attention mask for the negative prompt as well.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Chroma.Width;
        int height = request.Height ?? GenerationDefaults.Chroma.Height;
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;
        bool useCfg = cfgScale > 1.0f;

        // Pixel-space: pad dimensions up to the 16-px patch grid, crop the output back at the end.
        int patch = _transformer.PatchSize;
        int padWidth = PadUpTo(width, patch);
        int padHeight = PadUpTo(height, patch);

        // Img2img / inpaint: pixel-space means no VAE — the source image IS the clean sample. Validate
        // against the request resolution (the pad to the patch grid is internal; source + mask are padded
        // alongside the sample below) and handle the strength=0 short-circuit before any model work.
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        bool isImg2Img = request is ImageToImageRequest;
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
        Logs.Info($"Chroma Radiance {opMode}: {width}x{height} (padded {padWidth}x{padHeight}), " +
            $"{steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts with T5-XXL (identical text path to ChromaPipeline, including the trim) ──
        // Prompt-embedding cache: identical token ids reuse the previous gen's hidden states — the whole T5
        // phase (DiT evict + T5 preload + encode + free) vanishes for repeat prompts (seed-only changes).
        bool condHit = _cachedCond is not null
            && _cachedCondKey is not null && _cachedCondKey.AsSpan().SequenceEqual(promptTokenIdsT5);
        bool uncondHit = !useCfg || (_cachedUncond is not null
            && _cachedUncondKey is not null && _cachedUncondKey.AsSpan().SequenceEqual(negativePromptTokenIdsT5));
        Tensor condContext;
        Tensor? uncondContext = null;
        if (condHit && uncondHit)
        {
            condContext = _cachedCond!;
            if (useCfg) uncondContext = _cachedUncond;
            Logs.Info("[Radiance] prompt-embedding cache hit — T5 phase skipped");
        }
        else
        {
            Logs.Info("Encoding text with T5-XXL...");
            if (_ditResident)
            {
                // T5-XXL (~5 GB) cannot coexist with the resident ~9 GB DiT on smaller cards — evict for this
                // new-prompt generation and re-preload below.
                Backend.Sync();
                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }
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

            // Trim the padded context to Chroma's kept tokens (text_len + 1) instead of masking — EXACT (the
            // dropped rows are masked out of every attention by the transformer-side rule) and it makes all 57
            // SDPAs mask-free while shrinking every joint-sequence GEMM. See ChromaPipeline for the derivation.
            condContext = ChromaPipeline.TrimContextToKeptTokens(condContext, promptAttentionMaskT5);
            if (uncondContext is not null)
                uncondContext = ChromaPipeline.TrimContextToKeptTokens(uncondContext, negativeAttentionMaskT5);

            Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms " +
                $"(context trimmed to {condContext.Shape[1]}/{uncondContext?.Shape[1].ToString() ?? "-"} kept tokens)");

            // The trim host-materialized the contexts, so they survive activation reclaims; cache for reuse.
            _cachedCond?.Dispose();
            _cachedCond = condContext;
            _cachedCondKey = (int[])promptTokenIdsT5.Clone();
            if (useCfg)
            {
                _cachedUncond?.Dispose();
                _cachedUncond = uncondContext;
                _cachedUncondKey = (int[])negativePromptTokenIdsT5.Clone();
            }

            Backend.Sync();
            Backend.FreeWeights(_t5.EnumerateWeights());
            // Release the encoder phase's pool pages before the ~19 GB transformer preload — the resident
            // DiT leaves only a few GB of headroom for the denoise-loop activations.
            Backend.TrimMemoryPool();
        }

        // ── 2. Static-shift flow-match Euler scheduler (shift 1.0 — see config) ──
        TensorShape pixelShape = new TensorShape(1, 3, padHeight, padWidth);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Initial pixel sample (the "latent" IS the image) ──
        // T2I: pure noise scaled by initSigma. Img2img: source padded to the patch grid + AddNoise at
        // sigma[startStep] — no VAE encode, the pixels themselves are the clean sample. Masked inpaint
        // keeps the padded source + padded mask alive for per-step blending (pad fill 1.0 lets the
        // padded border denoise freely; it's cropped away at the end either way).
        Tensor pixels = TakeOrCreateNoise(request, pixelShape, seed);
        if (!pixels.Shape.Equals(pixelShape))
            throw new ArgumentException($"InitialNoise shape {pixels.Shape} != expected {pixelShape}.");
        Tensor? sourcePadded = null;
        Tensor? maskPadded = null;
        if (isImg2Img)
        {
            sourcePadded = PadPixels(((ImageToImageRequest)request).SourceImage, padHeight, padWidth, fill: 0f);
            Tensor noised = new Tensor(pixelShape, DType.F32);
            scheduler.AddNoise(noised, sourcePadded, pixels, startStep);
            pixels.Dispose();
            pixels = noised;
            if (isMaskedInpaint)
            {
                maskPadded = PadPixels(maskPixel!, padHeight, padWidth, fill: 1f);
            }
            else
            {
                sourcePadded.Dispose();
                sourcePadded = null;
            }
        }
        else
        {
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = new Tensor(pixelShape, DType.F32);
                Backend.Scale(scaled, pixels, initSigma);
                pixels.Dispose();
                pixels = scaled;
            }
        }

        // ── 4. Denoising loop ──
        // The transformer stays device-resident across generations; only a new-prompt T5 phase evicts it
        // (see the encode section above). On a card that cannot hold it, the 19+38 backbone blocks ride a
        // BlockStreamingController instead and only the shared set (approximator, context_embedder, conv
        // patchifier, NeRF head) stays resident — Radiance's ~9.3 GB fp8 DiT plus a pixel-space NeRF head whose
        // final conv alone asks for ~1 GB of workspace at 1024² does not fit a 12 GB card (measured 2026-07-28:
        // OOM at the head's Conv2D with 49 MB free after the blocks filled the card).
        BlockStreamingController? streamer = null;
        if (Backend.StreamingCache is not null)
        {
            IStreamingBlock[] blocks = new IStreamingBlock[_transformer.BlockCount];
            for (int b = 0; b < blocks.Length; b++) blocks[b] = _transformer.GetBlock(b);
            long totalBlockBytes = 0;
            foreach (IStreamingBlock block in blocks) totalBlockBytes += block.EstimatedWeightBytes;
            long sharedBytes = WeightBytes.Sum(_transformer.EnumerateSharedWeights());
            long reserve = EstimateActivationReserveBytes(
                padHeight, padWidth, _config.NerfHidden, patch, (int)condContext.Shape[1]) + sharedBytes;

            VramPlanner planner = new VramPlanner(Backend.StreamingCache, "ChromaRadiance", Backend);
            // Forced streaming has to displace a warm DiT before the planner measures, or the already-resident
            // short-circuit answers Resident and the setting does nothing on exactly the generations it was set for.
            if (planner.ShouldDisplaceResident(_ditResident))
            {
                Backend.Sync();
                Backend.FreeWeights(_transformer.EnumerateWeights());
                _ditResident = false;
            }
            PhasePlacement placement = planner.PlanPhase(
                "denoise", totalBlockBytes, reserve, alreadyResident: _ditResident, canStream: true);
            // Assigned unconditionally below (null on the resident branch): the transformer outlives this call,
            // so a hook left over from a previous streamed generation would keep calling a disposed controller.
            _transformer.BeforeBlockForward = null;
            if (placement == PhasePlacement.Resident)
            {
                Backend.PreloadWeights(_transformer.EnumerateWeights());
                _ditResident = true;
            }
            else
            {
                Backend.PreloadWeights(_transformer.EnumerateSharedWeights());
                long avail = Backend.StreamingCache.QueryAvailableWeightCacheBytes(reserve);
                // The widest block sizes the window: block 0 is a double-stream block (~2x a single-stream one),
                // so budgeting on it keeps the deepest prefetch safe for every block.
                long perBlock = blocks.Length > 0 ? blocks[0].EstimatedWeightBytes : 0;
                int prefetchAhead = perBlock > 0 ? Math.Clamp((int)(avail / perBlock) - 2, 0, 2) : 0;
                streamer = new BlockStreamingController(
                    Backend.StreamingCache, blocks, prefetchAhead: prefetchAhead, retainBehind: 0, backend: Backend);
                _transformer.BeforeBlockForward = streamer.BeforeBlockForward;
                streamer.Prime();
                _ditResident = false;
                // Radiance runs the full stack TWICE per step (cond + uncond can't batch — different text
                // lengths), so the per-step H2D bill is 2x totalBlockBytes.
                Logs.Info($"Chroma Radiance streaming: {blocks.Length} blocks, prefetchAhead={prefetchAhead}, " +
                    $"total ~{totalBlockBytes / (1024 * 1024)} MB{(useCfg ? " x2 passes/step" : "")}, " +
                    $"shared ~{sharedBytes / (1024 * 1024)} MB resident, reserve ~{reserve / (1024 * 1024)} MB");
            }
        }
        else
        {
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
        }

        Logs.Info("Starting Chroma Radiance denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int condTxtLen = (int)condContext.Shape[1];
        int uncondTxtLen = useCfg ? (int)uncondContext!.Shape[1] : 0;

        // Drain-free fast path (plain t2i / img2img): the pixel sample stays device-resident across the whole
        // loop — x0→velocity conversion runs as device elementwise ops and CFG combine + Euler is ONE in-place
        // CfgEulerStep (cond-anchored CFG maps onto the uncond-anchored kernel via guidance = cfg + 1). Masked
        // inpaint keeps the host branch — it must read/rebuild the sample on the host every step.
        bool drainFree = !isMaskedInpaint;

        // Sampler selection (2026-08-20). Chroma Radiance is flow-matching, so it had no user-selectable sampler at
        // all before this. The predictor converts each x0 pass to velocity and hands the RAW pair back, so the
        // sampler integrates in the same velocity domain the direct CfgEulerStep call used.
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, "Chroma Radiance",
            startsFromNoisedInit: startStep > 0);
        bool nonDefaultSampler = FlowMatchSampling.IsNonDefault(request.Scheduler);

        // The host/reference branch below does not consult the sampler at all, so a non-default selection there would
        // be silently dropped — the exact failure this whole change removes. Refuse by name instead.
        if (!drainFree && nonDefaultSampler)
        {
            throw new NotSupportedException(
                $"Sampler/schedule '{request.Scheduler}' runs only on Chroma Radiance's drain-free path, and this "
                + "generation fell back to the reference loop (masked inpaint). Drop the sampler selection, or "
                + "drop the feature that forced the fallback.");
        }

        // The predictor closure needs the timestep table, and `timesteps` is a ReadOnlySpan (a ref local) that a
        // lambda cannot capture. One small array copy per generation.
        float[] timestepTable = timesteps.ToArray();

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            if (drainFree)
            {
                DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(
                    PredictionType.FlowVelocity,
                    (x, s, stepIndex) =>
                    {
                        // On-schedule sigmas reuse the loop's own `timesteps[i]/1000` expression: the two are equal
                        // mathematically, but the F32 round trip through x1000 is not exact, and substituting one for
                        // the other would shift every existing Radiance generation by an ulp of conditioning.
                        float t = stepIndex < steps && s == scheduler.SigmaAt(stepIndex)
                            ? timestepTable[stepIndex] / 1000.0f : s;
                        // Model predicts x0; convert each pass to velocity BEFORE the CFG combine (ComfyUI order).
                        // The conversion is a function of the sample being evaluated, so it runs against `x` and this
                        // evaluation's own t — a sub-step's x0 residual belongs to ITS sample, not the step's.
                        (Tensor condX0, Tensor? uncondX0) = _transformer.ForwardPaired(
                            Backend, x, condContext, useCfg ? uncondContext : null, t, null, null);
                        Tensor condV = X0Prediction.ToVelocityDevice(Backend, condX0, x, t);
                        condX0.Dispose();
                        if (!useCfg)
                        {
                            return new DenoisePrediction(condV, condV);
                        }
                        Tensor uncondV = X0Prediction.ToVelocityDevice(Backend, uncondX0!, x, t);
                        uncondX0!.Dispose();
                        // Cond-anchored CFG (cond + s·(cond − uncond)) == uncond-anchored combine at guidance s+1,
                        // which is what both the fused kernel and SamplerMath.CombineCfg apply.
                        return new DenoisePrediction(condV, uncondV, cfgScale + 1.0f);
                    });
                if (i == startStep)
                {
                    sampler.Reset(pixels.Shape);
                }
                sampler.Step(Backend, pixels, predictor, i);
            }
            else
            {
                Tensor condX0 = _transformer.Forward(Backend, pixels, condContext, sigma, condTxtLen, null);
                Tensor velocity = X0Prediction.ToVelocity(condX0, pixels, sigma);
                condX0.Dispose();

                if (useCfg)
                {
                    Tensor uncondX0 = _transformer.Forward(Backend, pixels, uncondContext!, sigma, uncondTxtLen, null);
                    Tensor uncondV = X0Prediction.ToVelocity(uncondX0, pixels, sigma);
                    uncondX0.Dispose();

                    // VALIDATION-PENDING: Chroma-family cond-anchored CFG (cond + scale*(cond - uncond)) on velocity; verify vs reference.
                    Tensor combined = CfgHelper.ApplyCfgCondAnchored(velocity, uncondV, cfgScale);
                    uncondV.Dispose();
                    velocity.Dispose();
                    velocity = combined;
                }

                Tensor newPixels = new Tensor(pixelShape, DType.F32);
                scheduler.Step(newPixels, velocity, pixels, i);
                velocity.Dispose();
                pixels.Dispose();
                pixels = newPixels;
            }

            // Masked-inpaint blend in pixel space: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source at the next step's sigma.
            // Final step blends with the clean source — no further denoising follows.
            if (maskPadded is not null && sourcePadded is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(pixelShape, seed + nextStep);
                    noisedSource = new Tensor(pixelShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourcePadded, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourcePadded;
                }
                MaskBlendUtilities.BlendChannelsInPlace(pixels, noisedSource, maskPadded);
                if (noisedSource != sourcePadded) noisedSource.Dispose();
            }

            // retainBehind:0 frees every block through cuMemFreeAsync, and HARTSY_MEMPOOL_KEEP holds those bytes
            // reserved — without a per-step trim the pool grows by roughly a block per step until it owns the card.
            streamer?.TrimAfterStep();

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            // Pixel-space preview: the in-flight sample is already an RGB image — no unpack needed. On the
            // drain-free path a preview read is a D2H sync of the device-resident sample, so previews emit
            // every 4th step (and not the final step — the finished image follows immediately).
            bool emitPreview = onProgress is not null && (!drainFree || ((i - startStep) % 4 == 3 && i != steps - 1));
            if (emitPreview)
            {
                onProgress!.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                {
                    Latent = pixels,
                    LatentArch = LatentArchitecture.ChromaRadiance,
                });
            }
            else
            {
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
            }
        }

        // condContext/uncondContext are cross-generation caches — not disposed here. The transformer stays
        // resident (freed only by a new-prompt T5 phase or pipeline disposal).
        if (streamer is not null)
        {
            // Nothing of the DiT survives a streamed generation: the sliding window's blocks go back, and the
            // shared set goes with them so the next generation re-plans against a clean card (its T5 phase needs
            // ~5 GB). _ditResident is already false, so nothing claims these are still up.
            _transformer.BeforeBlockForward = null;
            streamer.EvictAll();
            streamer.Dispose();
            streamer = null;
            Backend.FreeWeights(_transformer.EnumerateSharedWeights());
        }
        sourcePadded?.Dispose();
        maskPadded?.Dispose();
        Backend.Sync();

        // ── 5. Crop padding (if any) and convert straight to RGB bytes — no VAE ──
        if (padWidth != width || padHeight != height)
        {
            Tensor cropped = CropPixels(pixels, height, width);
            pixels.Dispose();
            pixels = cropped;
        }

        // Recomposite for masked inpaint: paste the denoised pixels over the source where mask=1.
        // Pixel space has no VAE round-trip drift, but the final blend guarantees the unmasked
        // region is byte-exact to the source (consistent with the latent-space pipelines).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(pixels, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(pixels);
        pixels.Dispose();

        sw.Stop();
        Logs.Info($"Chroma Radiance {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    private static int PadUpTo(int n, int multiple)
    {
        int rem = n % multiple;
        return rem == 0 ? n : n + (multiple - rem);
    }

    /// <summary>Activations + workspace the denoise phase needs alongside the weights, for <see cref="VramPlanner.PlanPhase"/>.</summary>
    /// <remarks>Radiance is unusual in that the NeRF pixel head, not the backbone, dominates: at 1024² its
    /// per-pixel embedding, F16 GLU stream and <c>[1, nerfHidden, H, W]</c> feature map are each ~268 MB of F32,
    /// and the final pixel conv's workspace was the allocation that OOM'd a 12 GB card (1024 MB requested with
    /// 49 MB free). Eight such planes plus the conv workspace cover the head; the block loop's own joint-sequence
    /// F16 buffers are comparatively small (~25 MB each at 1024²) but there are many live at once.</remarks>
    private static long EstimateActivationReserveBytes(int padHeight, int padWidth, int nerfHidden, int patch, int txtSeqLen)
    {
        long pixels = (long)padHeight * padWidth;
        long nerfBytes = (pixels * nerfHidden * sizeof(float) * 8) + (pixels * 9 * sizeof(float));
        long jointSeqLen = (pixels / ((long)patch * patch)) + txtSeqLen;
        long blockBytes = jointSeqLen * ChromaBlockStreamHiddenSize * sizeof(ushort) * 16;
        return nerfBytes + blockBytes;
    }

    /// <summary>Chroma's hidden size (3072) — used only to size the streaming planner's activation reserve.</summary>
    private const int ChromaBlockStreamHiddenSize = 3072;

    /// <summary>Pads a <c>[1, C, H, W]</c> pixel tensor to <c>[1, C, padH, padW]</c> (top-left anchored, matching <see cref="CropPixels"/>) with <paramref name="fill"/> in the new bottom/right border. Source images pad with 0 (mid-gray in [-1, 1]); masks pad with 1 so the cropped-away border denoises freely.</summary>
    private static Tensor PadPixels(Tensor src, int padH, int padW, float fill)
    {
        int channels = (int)src.Shape[1];
        int h = (int)src.Shape[2];
        int w = (int)src.Shape[3];
        if (h == padH && w == padW)
        {
            Tensor clone = new Tensor(src.Shape, DType.F32);
            long bytes = src.Shape.ElementCount * sizeof(float);
            Buffer.MemoryCopy((float*)src.DataPointer, (float*)clone.DataPointer, bytes, bytes);
            return clone;
        }
        Tensor output = new Tensor(new TensorShape(1, channels, padH, padW), DType.F32);
        float* srcPtr = (float*)src.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            long srcPlane = (long)c * h * w;
            long dstPlane = (long)c * padH * padW;
            for (int y = 0; y < padH; y++)
            {
                float* dstRow = dstPtr + dstPlane + (long)y * padW;
                if (y < h)
                {
                    long rowBytes = (long)w * sizeof(float);
                    Buffer.MemoryCopy(srcPtr + srcPlane + (long)y * w, dstRow, rowBytes, rowBytes);
                    for (int x = w; x < padW; x++) dstRow[x] = fill;
                }
                else
                {
                    for (int x = 0; x < padW; x++) dstRow[x] = fill;
                }
            }
        }
        return output;
    }

    /// <summary>Crops a [1, 3, padH, padW] pixel tensor back to the requested [1, 3, H, W] (top-left anchored, matching ComfyUI's <c>[:, :, :h, :w]</c> un-pad).</summary>
    private static Tensor CropPixels(Tensor padded, int height, int width)
    {
        int padHeight = (int)padded.Shape[2];
        int padWidth = (int)padded.Shape[3];
        Tensor output = new Tensor(new TensorShape(1, 3, height, width), DType.F32);

        float* srcPtr = (float*)padded.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            long srcPlane = (long)c * padHeight * padWidth;
            long dstPlane = (long)c * height * width;
            for (int y = 0; y < height; y++)
            {
                long rowBytes = (long)width * sizeof(float);
                Buffer.MemoryCopy(
                    srcPtr + srcPlane + (long)y * padWidth,
                    dstPtr + dstPlane + (long)y * width,
                    rowBytes, rowBytes);
            }
        }
        return output;
    }
}
