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

/// <summary>Chroma text-to-image pipeline. Identical to Flux pipeline but uses standard classifier-free guidance (dual forward pass with negative prompt) instead of Flux Dev's embedded guidance. Reuses FluxTransformer directly.</summary>
public sealed unsafe class ChromaPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly T5TextEncoder _t5;
    private readonly FluxTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly ChromaConfig _config;
    private int _disposed;

    /// <summary>Creates a new Chroma pipeline with all components pre-loaded.</summary>
    public ChromaPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, ChromaConfig config)
    {
        _backend = backend;
        _clipL = clipL;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized input with standard CFG.</summary>
    /// <param name="promptTokenIdsL">Prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="negativePromptTokenIdsL">Negative prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="promptEosPositionL">EOS token position in prompt for CLIP-L.</param>
    /// <param name="negativeEosPositionL">EOS token position in negative prompt for CLIP-L.</param>
    /// <param name="promptTokenIdsT5">Prompt token IDs for T5-XXL [seqLen].</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs for T5 [seqLen].</param>
    /// <param name="promptAttentionMaskT5">T5 attention mask for prompt. Null = attend all.</param>
    /// <param name="negativeAttentionMaskT5">T5 attention mask for negative prompt. Null = attend all.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int[] negativePromptTokenIdsL,
        int promptEosPositionL,
        int negativeEosPositionL,
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
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

        Logs.Info($"Chroma: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text (positive + negative) ─────────────────────────
        Logs.Info("Encoding text with CLIP-L (pooled) + T5-XXL (per-token)...");

        // Positive prompt
        int[][] batchTokenIdsL = [promptTokenIdsL];
        Tensor clipLHidden = _clipL.Encode(_backend, batchTokenIdsL);
        Tensor condClipPooled = ExtractEosHiddenState(clipLHidden, promptEosPositionL);
        clipLHidden.Dispose();

        int[][] batchTokenIdsT5 = [promptTokenIdsT5];
        int[][]? batchMaskT5 = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor condT5Embeddings = _t5.Encode(_backend, batchTokenIdsT5, batchMaskT5);

        // Negative prompt
        int[][] negBatchTokenIdsL = [negativePromptTokenIdsL];
        Tensor negClipLHidden = _clipL.Encode(_backend, negBatchTokenIdsL);
        Tensor uncondClipPooled = ExtractEosHiddenState(negClipLHidden, negativeEosPositionL);
        negClipLHidden.Dispose();

        int[][] negBatchTokenIdsT5 = [negativePromptTokenIdsT5];
        int[][]? negBatchMaskT5 = negativeAttentionMaskT5 is not null ? [negativeAttentionMaskT5] : null;
        Tensor uncondT5Embeddings = _t5.Encode(_backend, negBatchTokenIdsT5, negBatchMaskT5);

        int txtSeqLen = (int)condT5Embeddings.Shape[1];
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Create initial noise ──────────────────────────────────────
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);

        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;

        // TODO: Reuse FluxPipeline.PackLatent once made internal/shared
        Tensor packedLatent = PackLatent(noise, latentH, latentW);
        noise.Dispose();

        // ── 3. Set up scheduler ──────────────────────────────────────────
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);
            Tensor scaled = new Tensor(packedShape, DType.F32);
            _backend.Scale(scaled, packedLatent, initSigma);
            packedLatent.Dispose();
            packedLatent = scaled;
        }

        // ── 4. Denoising loop with standard CFG ──────────────────────────
        Logs.Info("Starting Chroma denoising loop (standard CFG)...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Unconditional forward pass
            Tensor uncondVelocity = _transformer.Forward(
                _backend, packedLatent, uncondT5Embeddings, sigma,
                uncondClipPooled, 0.0f, txtSeqLen, hPacked, wPacked);

            // Conditional forward pass
            Tensor condVelocity = _transformer.Forward(
                _backend, packedLatent, condT5Embeddings, sigma,
                condClipPooled, 0.0f, txtSeqLen, hPacked, wPacked);

            // CFG: output = uncond + scale * (cond - uncond)
            Tensor guidedVelocity = ApplyCfg(uncondVelocity, condVelocity, cfgScale);
            uncondVelocity.Dispose();
            condVelocity.Dispose();

            // Scheduler step
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, 64);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            scheduler.Step(newLatent, guidedVelocity, packedLatent, i);
            guidedVelocity.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        condClipPooled.Dispose();
        uncondClipPooled.Dispose();
        condT5Embeddings.Dispose();
        uncondT5Embeddings.Dispose();

        // ── 5. Unpack + VAE decode ───────────────────────────────────────
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();

        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, unpackedLatent);
        unpackedLatent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Chroma image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Applies classifier-free guidance: output = uncond + scale * (cond - uncond).</summary>
    private Tensor ApplyCfg(Tensor uncond, Tensor cond, float cfgScale)
    {
        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uncPtr = (float*)uncond.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)uncond.ElementCount;

        for (int i = 0; i < count; i++)
            outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);

        return output;
    }

    // TODO: Extract PackLatent/UnpackLatent to shared utility (duplicated from FluxPipeline)
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

        return packed;
    }

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
                dstPtr[dstOffset + d] = srcPtr[srcOffset + d];
        }

        return pooled;
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
