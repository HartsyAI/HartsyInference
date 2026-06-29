using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>SD3 text-to-image and image-to-image pipeline. Orchestrates triple text encoding (CLIP-L + CLIP-G + T5-XXL) → MMDiT denoising with flow matching → VAE decode → RGB image output. T5 is optional for reduced VRAM usage.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>). Requires a <see cref="VaeEncoder"/> on the constructor. Setting <see cref="ImageToImageRequest.Mask"/> additionally enables blend-on-vanilla inpaint, identical to the SDXL/Flux paths but on SD3's 16-channel NCHW latent.</para></summary>
public sealed unsafe class Sd3Pipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly T5TextEncoder? _t5;
    private readonly Sd3Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly float _schedulerShift;

    /// <summary>Creates a new SD3 pipeline with all components pre-loaded. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public Sd3Pipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder? t5, Sd3Transformer transformer, VaeDecoder vaeDecoder,
        float schedulerShift = 3.0f)
        : this(backend, clipL, clipG, t5, transformer, vaeDecoder, vaeEncoder: null, schedulerShift)
    {
    }

    /// <summary>Creates a new SD3 pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public Sd3Pipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder? t5, Sd3Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, float schedulerShift = 3.0f)
        : base(backend)
    {
        _clipL = clipL;
        _clipG = clipG;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized input for all three text encoders.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int[] negativePromptTokenIdsL,
        int[] promptTokenIdsG,
        int[] negativePromptTokenIdsG,
        int promptEosPositionL,
        int negativeEosPositionL,
        int promptEosPositionG,
        int negativeEosPositionG,
        int[]? promptTokenIdsT5,
        int[]? negativePromptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        int[]? negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Sd35.Resolve(request);
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

        string mode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                    : isImg2Img ? $"img2img (start={startStep}/{steps})"
                    : "txt2img";
        Logs.Info($"SD3 {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with triple encoders ─────────────────────────
        Logs.Info("Encoding text with CLIP-L, CLIP-G, T5...");

        // Bulk-upload T5 weights once (~5 GB on T5-XXL) so the encoder's many kernels don't
        // each pay a per-op cache-miss H2D transfer. Backends without a weight cache no-op.
        // Pairs with the FreeWeights below the VAE handoff. CLIP-L/G are tiny and we let them
        // lazy-cache (no matching free either — they stay resident across generations).
        if (_t5 is not null) Backend.PreloadWeights(_t5.EnumerateWeights());

        bool useCfg = cfgScale > 1.0f;

        // Encode positive prompt
        // SD3 is specified against the penultimate CLIP layer; request.ClipSkip overrides (2 = penultimate).
        int clipSkip = request.ClipSkip ?? 2;
        (Tensor condContext, Tensor condPooled) = EncodePrompt(
            promptTokenIdsL, promptTokenIdsG, promptTokenIdsT5,
            promptEosPositionL, promptEosPositionG,
            promptAttentionMaskT5, clipSkip);

        // Project context through transformer's context_embedder
        Tensor condProjected = _transformer.ProjectContext(Backend, condContext);
        condContext.Dispose();

        Tensor? uncondProjected = null;
        Tensor? uncondPooled = null;

        if (useCfg)
        {
            // Encode negative prompt
            (Tensor negContext, Tensor negPooled) = EncodePrompt(
                negativePromptTokenIdsL, negativePromptTokenIdsG, negativePromptTokenIdsT5,
                negativeEosPositionL, negativeEosPositionG,
                negativeAttentionMaskT5, clipSkip);

            uncondProjected = _transformer.ProjectContext(Backend, negContext);
            negContext.Dispose();
            uncondPooled = negPooled;
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Set up flow-match scheduler ──────────────────────────────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Build initial latent ─────────────────────────────────────
        // T2I: noise scaled by initSigma. Img2img: VAE-encode source + AddNoise at sigma[startStep].
        // Masked inpaint: keep clean source latent + downsampled mask alive for per-step blend.
        Tensor latent;
        Tensor? sourceLatent = null;
        Tensor? latentMask = null;
        if (request is ImageToImageRequest img2imgInit)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            sourceLatent = _vaeEncoder!.Encode(Backend, img2imgInit.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            noise.Dispose();

            if (isMaskedInpaint)
            {
                latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(maskPixel!, latentH, latentW);
            }
            else
            {
                sourceLatent.Dispose();
                sourceLatent = null;
            }
        }
        else
        {
            latent = SeedGenerator.CreateNoise(latentShape, seed);
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = new Tensor(latentShape, DType.F32);
                Backend.Scale(scaled, latent, initSigma);
                latent.Dispose();
                latent = scaled;
            }
        }

        // ── 4. Denoising loop ───────────────────────────────────────────
        // Bulk-upload transformer weights before the denoise loop. The transformer is touched
        // every step, every block — without preload, the first step would pay cache-miss
        // overhead for every parameter access. FreeWeights below the VAE handoff cycles the
        // VRAM. No-op on backends without a weight cache.
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting SD3 denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            Tensor noisePred;
            if (useCfg)
            {
                noisePred = ClassifierFreeGuidanceStep(
                    latent, t, condProjected, condPooled, uncondProjected!, uncondPooled!, cfgScale);
            }
            else
            {
                noisePred = _transformer.Forward(Backend, latent, t, condProjected, condPooled);
            }

            // Scheduler step
            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            // Masked-inpaint blend: replace unmasked region with re-noised source at the
            // next step's sigma. Same formulation as SDXL — only the channel count differs.
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
                LatentArch = LatentArchitecture.Sd3,
            });
        }

        condProjected.Dispose();
        condPooled.Dispose();
        uncondProjected?.Dispose();
        uncondPooled?.Dispose();
        sourceLatent?.Dispose();
        latentMask?.Dispose();

        Sd3Transformer.DumpFinalLatent(latent);

        // Free transformer + text encoder weights from GPU now that denoising is done.
        // The VAE im2col buffers at 512×512+ are large (hundreds of MB to several GB),
        // and on a 12 GB card we OOM during VAE decode if the transformer is still
        // resident. Mirrors PHASE_3_DEVIATIONS #18 (UNet eviction before VAE for SDXL/Flux).
        // Backends without a weight cache (CPU/Vulkan) treat this as a no-op.
        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        if (_t5 is not null) Backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. VAE decode ───────────────────────────────────────────────
        // Tiled decode: caps im2col workspace at ~2.4 GB per tile. Internal fast-path
        // skips tiling when the latent fits in a single tile, so small images pay no overhead.
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.DecodeTiled(Backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 6. Pixel-space recomposite for masked inpaint ──────────────
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        // ── 7. Convert to RGB bytes ─────────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SD3 {mode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Encodes a single prompt through all three text encoders and combines the results.</summary>
    private (Tensor context, Tensor pooled) EncodePrompt(
        int[] tokenIdsL, int[] tokenIdsG, int[]? tokenIdsT5,
        int eosPositionL, int eosPositionG,
        int[]? attentionMaskT5, int clipSkip = 2)
    {
        int seqLenClip = tokenIdsL.Length;

        // CLIP-L: penultimate hidden [1, 77, 768] + pooled [1, 768]
        int[][] batchTokenIdsL = [tokenIdsL];
        int[] eosPositionsL = [eosPositionL];
        (Tensor clipLHidden, Tensor? clipLPooled) = _clipL.EncodePenultimate(Backend, batchTokenIdsL, eosPositionsL, clipSkip);

        // CLIP-G: penultimate hidden [1, 77, 1280] + pooled [1, 1280]
        int[][] batchTokenIdsG = [tokenIdsG];
        int[] eosPositionsG = [eosPositionG];
        (Tensor clipGHidden, Tensor? clipGPooled) = _clipG.EncodePenultimate(Backend, batchTokenIdsG, eosPositionsG, clipSkip);

        // Combine pooled: concat(clip_l_pooled, clip_g_pooled, dim=-1) → [1, 2048]
        Tensor pooled = ConcatPooled(clipLPooled!, clipGPooled!);
        clipLPooled?.Dispose();
        clipGPooled?.Dispose();

        // Combine hidden: concat(clip_l_hidden, clip_g_hidden, dim=-1) → [1, 77, 2048]
        Tensor lgHidden = CfgHelper.ConcatLastDim(clipLHidden, clipGHidden);
        clipLHidden.Dispose();
        clipGHidden.Dispose();

        // Pad to 4096: [1, 77, 2048] → [1, 77, 4096]
        int lgDim = (int)lgHidden.Shape[2];
        int targetDim = 4096;
        Tensor lgPadded = PadLastDim(lgHidden, lgDim, targetDim);
        lgHidden.Dispose();

        // T5 encoding
        Tensor t5Hidden;
        if (_t5 is not null && tokenIdsT5 is not null)
        {
            int[][] batchT5 = [tokenIdsT5];
            int[][]? batchMask = attentionMaskT5 is not null ? [attentionMaskT5] : null;
            t5Hidden = _t5.Encode(Backend, batchT5, batchMask);
        }
        else
        {
            // T5 dropout: zero tensor [1, seqLen, 4096]
            int t5SeqLen = tokenIdsT5?.Length ?? seqLenClip;
            TensorShape t5Shape = new TensorShape(1, t5SeqLen, targetDim);
            t5Hidden = new Tensor(t5Shape, DType.F32);
            Backend.Fill(t5Hidden, 0.0f);
        }

        // Concat along sequence: [1, 77, 4096] + [1, 77, 4096] → [1, 154, 4096]
        Tensor context = ConcatAlongSeqDim(lgPadded, t5Hidden);
        lgPadded.Dispose();
        t5Hidden.Dispose();

        return (context, pooled);
    }

    /// <summary>Runs classifier-free guidance: noise_pred = uncond + cfg_scale * (cond - uncond).</summary>
    private Tensor ClassifierFreeGuidanceStep(
        Tensor latent, float timestep,
        Tensor condContext, Tensor condPooled,
        Tensor uncondContext, Tensor uncondPooled,
        float cfgScale)
    {
        Tensor uncondNoise = _transformer.Forward(Backend, latent, timestep, uncondContext, uncondPooled);
        Tensor condNoise = _transformer.Forward(Backend, latent, timestep, condContext, condPooled);
        Tensor output = CfgHelper.ApplyCfg(uncondNoise, condNoise, cfgScale);
        uncondNoise.Dispose();
        condNoise.Dispose();
        return output;
    }

    /// <summary>Concatenates two [B, S1, D] and [B, S2, D] tensors along the sequence dimension → [B, S1+S2, D].</summary>
    private static Tensor ConcatAlongSeqDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqA = (int)a.Shape[1];
        int seqB = (int)b.Shape[1];
        int dim = (int)a.Shape[2];
        int seqOut = seqA + seqB;

        TensorShape outShape = new TensorShape(batch, seqOut, dim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            // Copy all of tensor a for this batch
            int aSrcOffset = bIdx * seqA * dim;
            int aDstOffset = bIdx * seqOut * dim;
            Buffer.MemoryCopy(aPtr + aSrcOffset, outPtr + aDstOffset, seqA * dim * sizeof(float), seqA * dim * sizeof(float));

            // Copy all of tensor b for this batch
            int bSrcOffset = bIdx * seqB * dim;
            int bDstOffset = bIdx * seqOut * dim + seqA * dim;
            Buffer.MemoryCopy(bPtr + bSrcOffset, outPtr + bDstOffset, seqB * dim * sizeof(float), seqB * dim * sizeof(float));
        }

        return output;
    }

    /// <summary>Pads the last dimension with zeros: [B, S, currentDim] → [B, S, targetDim].</summary>
    private static Tensor PadLastDim(Tensor input, int currentDim, int targetDim)
    {
        if (currentDim == targetDim)
            return input;

        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];

        TensorShape outShape = new TensorShape(batch, seqLen, targetDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Zero the entire output first, then copy input data
        int totalElements = batch * seqLen * targetDim;
        for (int i = 0; i < totalElements; i++)
            outPtr[i] = 0.0f;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * currentDim;
                int outOffset = (b * seqLen + s) * targetDim;
                Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, currentDim * sizeof(float), currentDim * sizeof(float));
            }
        }

        return output;
    }

    /// <summary>Concatenates two pooled tensors [B, D1] and [B, D2] along the last dimension → [B, D1+D2].</summary>
    private static Tensor ConcatPooled(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int dimA = (int)a.Shape[1];
        int dimB = (int)b.Shape[1];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            Buffer.MemoryCopy(aPtr + bIdx * dimA, outPtr + bIdx * dimOut, dimA * sizeof(float), dimA * sizeof(float));
            Buffer.MemoryCopy(bPtr + bIdx * dimB, outPtr + bIdx * dimOut + dimA, dimB * sizeof(float), dimB * sizeof(float));
        }

        return output;
    }
}
