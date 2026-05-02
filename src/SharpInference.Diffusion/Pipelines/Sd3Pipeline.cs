using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>SD3 text-to-image pipeline. Orchestrates triple text encoding (CLIP-L + CLIP-G + T5-XXL) → MMDiT denoising with flow matching → VAE decode → RGB image output. T5 is optional for reduced VRAM usage.</summary>
public sealed unsafe class Sd3Pipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly T5TextEncoder? _t5;
    private readonly Sd3Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly float _schedulerShift;
    private int _disposed;

    /// <summary>Creates a new SD3 pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipL">CLIP ViT-L/14 text encoder (text_encoder). Must have text_projection for pooled output.</param>
    /// <param name="clipG">OpenCLIP ViT-bigG/14 text encoder (text_encoder_2). Must have text_projection for pooled output.</param>
    /// <param name="t5">T5-XXL text encoder (text_encoder_3). Null to skip T5 conditioning (reduced VRAM).</param>
    /// <param name="transformer">SD3 MMDiT transformer (loaded with Sd3Config).</param>
    /// <param name="vaeDecoder">VAE decoder (configured with VaeConfig.Sd3).</param>
    /// <param name="schedulerShift">Flow-match scheduler shift. Default: 3.0 for SD3 Medium.</param>
    public Sd3Pipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG,
        T5TextEncoder? t5, Sd3Transformer transformer, VaeDecoder vaeDecoder,
        float schedulerShift = 3.0f)
    {
        _backend = backend;
        _clipL = clipL;
        _clipG = clipG;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _schedulerShift = schedulerShift;
    }

    /// <summary>Generates an image from pre-tokenized input for all three text encoders.</summary>
    /// <param name="promptTokenIdsL">Prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="negativePromptTokenIdsL">Negative prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="promptTokenIdsG">Prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="negativePromptTokenIdsG">Negative prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="promptEosPositionL">EOS token position in prompt for CLIP-L (for pooled output).</param>
    /// <param name="negativeEosPositionL">EOS token position in negative prompt for CLIP-L.</param>
    /// <param name="promptEosPositionG">EOS token position in prompt for CLIP-G (for pooled output).</param>
    /// <param name="negativeEosPositionG">EOS token position in negative prompt for CLIP-G.</param>
    /// <param name="promptTokenIdsT5">Prompt token IDs for T5-XXL [seqLen]. Null if T5 is not available.</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs for T5. Null if T5 is not available.</param>
    /// <param name="promptAttentionMaskT5">T5 attention mask for prompt (1=attend, 0=pad). Null = attend all.</param>
    /// <param name="negativeAttentionMaskT5">T5 attention mask for negative prompt. Null = attend all.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
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

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"SD3: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text with triple encoders ─────────────────────────
        Logs.Info("Encoding text with CLIP-L, CLIP-G, T5...");

        bool useCfg = cfgScale > 1.0f;

        // Encode positive prompt
        (Tensor condContext, Tensor condPooled) = EncodePrompt(
            promptTokenIdsL, promptTokenIdsG, promptTokenIdsT5,
            promptEosPositionL, promptEosPositionG,
            promptAttentionMaskT5);

        // Project context through transformer's context_embedder
        Tensor condProjected = _transformer.ProjectContext(_backend, condContext);
        condContext.Dispose();

        Tensor? uncondProjected = null;
        Tensor? uncondPooled = null;

        if (useCfg)
        {
            // Encode negative prompt
            (Tensor negContext, Tensor negPooled) = EncodePrompt(
                negativePromptTokenIdsL, negativePromptTokenIdsG, negativePromptTokenIdsT5,
                negativeEosPositionL, negativeEosPositionG,
                negativeAttentionMaskT5);

            uncondProjected = _transformer.ProjectContext(_backend, negContext);
            negContext.Dispose();
            uncondPooled = negPooled;
        }

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Create initial noise latent [1, 16, latentH, latentW] ────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // ── 3. Set up flow-match scheduler ──────────────────────────────
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(_schedulerShift);
        scheduler.SetTimesteps(steps);

        // Scale initial noise by sigma[0]
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // ── 4. Denoising loop ───────────────────────────────────────────
        Logs.Info("Starting SD3 denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
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
                noisePred = _transformer.Forward(_backend, latent, t, condProjected, condPooled);
            }

            // Scheduler step
            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condProjected.Dispose();
        condPooled.Dispose();
        uncondProjected?.Dispose();
        uncondPooled?.Dispose();

        Sd3Transformer.DumpFinalLatent(latent);

        // Free transformer + text encoder weights from GPU now that denoising is done.
        // The VAE im2col buffers at 512×512+ are large (hundreds of MB to several GB),
        // and on a 12 GB card we OOM during VAE decode if the transformer is still
        // resident. Mirrors PHASE_3_DEVIATIONS #18 (UNet eviction before VAE for SDXL/Flux).
        // Backends without a weight cache (CPU/Vulkan) treat this as a no-op.
        _backend.Sync();
        _backend.FreeWeights(_transformer.EnumerateWeights());
        if (_t5 is not null) _backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. VAE decode ───────────────────────────────────────────────
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 6. Convert to RGB bytes ─────────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SD3 image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Encodes a single prompt through all three text encoders and combines the results.</summary>
    private (Tensor context, Tensor pooled) EncodePrompt(
        int[] tokenIdsL, int[] tokenIdsG, int[]? tokenIdsT5,
        int eosPositionL, int eosPositionG,
        int[]? attentionMaskT5)
    {
        int seqLenClip = tokenIdsL.Length;

        // CLIP-L: penultimate hidden [1, 77, 768] + pooled [1, 768]
        int[][] batchTokenIdsL = [tokenIdsL];
        int[] eosPositionsL = [eosPositionL];
        (Tensor clipLHidden, Tensor? clipLPooled) = _clipL.EncodePenultimate(_backend, batchTokenIdsL, eosPositionsL);

        // CLIP-G: penultimate hidden [1, 77, 1280] + pooled [1, 1280]
        int[][] batchTokenIdsG = [tokenIdsG];
        int[] eosPositionsG = [eosPositionG];
        (Tensor clipGHidden, Tensor? clipGPooled) = _clipG.EncodePenultimate(_backend, batchTokenIdsG, eosPositionsG);

        // Combine pooled: concat(clip_l_pooled, clip_g_pooled, dim=-1) → [1, 2048]
        Tensor pooled = ConcatPooled(clipLPooled!, clipGPooled!);
        clipLPooled?.Dispose();
        clipGPooled?.Dispose();

        // Combine hidden: concat(clip_l_hidden, clip_g_hidden, dim=-1) → [1, 77, 2048]
        Tensor lgHidden = ConcatAlongLastDim(clipLHidden, clipGHidden);
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
            t5Hidden = _t5.Encode(_backend, batchT5, batchMask);
        }
        else
        {
            // T5 dropout: zero tensor [1, seqLen, 4096]
            int t5SeqLen = tokenIdsT5?.Length ?? seqLenClip;
            TensorShape t5Shape = new TensorShape(1, t5SeqLen, targetDim);
            t5Hidden = new Tensor(t5Shape, DType.F32);
            _backend.Fill(t5Hidden, 0.0f);
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
        Tensor uncondNoise = _transformer.Forward(_backend, latent, timestep, uncondContext, uncondPooled);
        Tensor condNoise = _transformer.Forward(_backend, latent, timestep, condContext, condPooled);

        // CFG: output = uncond + scale * (cond - uncond)
        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* uncPtr = (float*)uncondNoise.DataPointer;
        float* conPtr = (float*)condNoise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)latent.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);
        }

        uncondNoise.Dispose();
        condNoise.Dispose();

        return output;
    }

    /// <summary>Concatenates two [B, seqLen, D1] and [B, seqLen, D2] tensors along the last dimension → [B, seqLen, D1+D2].</summary>
    private static Tensor ConcatAlongLastDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, seqLen, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;

                Buffer.MemoryCopy(aPtr + aOffset, outPtr + outOffset, dimA * sizeof(float), dimA * sizeof(float));
                Buffer.MemoryCopy(bPtr + bOffset, outPtr + outOffset + dimA, dimB * sizeof(float), dimB * sizeof(float));
            }
        }

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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components (shared resources).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
