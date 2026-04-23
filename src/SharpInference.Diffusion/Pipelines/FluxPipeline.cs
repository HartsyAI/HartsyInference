using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Flux text-to-image pipeline. Orchestrates CLIP-L pooled + T5-XXL text encoding → FluxTransformer denoising with flow matching → VAE decode → RGB image output. Supports Dev (guidance embedding) and Schnell (distilled, 1-4 steps) modes.</summary>
public sealed unsafe class FluxPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly T5TextEncoder _t5;
    private readonly FluxTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly FluxConfig _config;
    private int _disposed;

    /// <summary>Creates a new Flux pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipL">CLIP ViT-L/14 text encoder (pooled output only).</param>
    /// <param name="t5">T5-XXL text encoder (per-token embeddings).</param>
    /// <param name="transformer">Flux transformer (configured with FluxConfig).</param>
    /// <param name="vaeDecoder">VAE decoder (configured with VaeConfig.Flux).</param>
    /// <param name="config">Flux configuration (Dev or Schnell).</param>
    public FluxPipeline(IBackend backend, ClipTextEncoder clipL, T5TextEncoder t5,
        FluxTransformer transformer, VaeDecoder vaeDecoder, FluxConfig config)
    {
        _backend = backend;
        _clipL = clipL;
        _t5 = t5;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized input.</summary>
    /// <param name="promptTokenIdsL">Prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="promptEosPositionL">EOS token position in prompt for CLIP-L (for pooled output).</param>
    /// <param name="promptTokenIdsT5">Prompt token IDs for T5-XXL [seqLen].</param>
    /// <param name="promptAttentionMaskT5">T5 attention mask (1=attend, 0=pad). Null = attend all.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="guidanceScale">Guidance scale for Dev (embedded via MLP). Ignored for Schnell. Default: 3.5.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int promptEosPositionL,
        int[] promptTokenIdsT5,
        int[]? promptAttentionMaskT5,
        TextToImageRequest request,
        float guidanceScale = 3.5f,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;

        string mode = _config.GuidanceEmbed ? "Dev" : "Schnell";
        Logs.Info($"Flux ({mode}): Generating {request.Width}x{request.Height} image, {steps} steps, guidance={guidanceScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode text ─────────────────────────────────────────────
        Logs.Info("Encoding text with CLIP-L (pooled) + T5-XXL (per-token)...");

        // CLIP-L: full forward → extract EOS hidden state as pooled output [B, 768]
        int[][] batchTokenIdsL = [promptTokenIdsL];
        Tensor clipLHidden = _clipL.Encode(_backend, batchTokenIdsL);
        Tensor clipPooled = ExtractEosHiddenState(clipLHidden, promptEosPositionL);
        clipLHidden.Dispose();

        // T5-XXL: per-token embeddings [B, seqLen, 4096]
        int[][] batchTokenIdsT5 = [promptTokenIdsT5];
        int[][]? batchMaskT5 = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor t5Embeddings = _t5.Encode(_backend, batchTokenIdsT5, batchMaskT5);

        int txtSeqLen = (int)t5Embeddings.Shape[1];
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (T5 seqLen={txtSeqLen})");

        // ── 2. Create initial noise latent [1, 16, latentH, latentW] ──
        TensorShape latentShape = new TensorShape(1, 16, latentH, latentW);
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);

        // Pack latent: [1, 16, H, W] → [1, H/2*W/2, 64]
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int imgSeqLen = hPacked * wPacked;

        Tensor packedLatent = PackLatent(noise, latentH, latentW);
        noise.Dispose();

        // ── 3. Set up dynamic flow-match scheduler ────────────────────
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imgSeqLen);
        scheduler.SetTimesteps(steps);

        // Scale initial noise by sigma[0]
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            TensorShape packedShape = new TensorShape(1, imgSeqLen, 64);
            Tensor scaled = new Tensor(packedShape, DType.F32);
            _backend.Scale(scaled, packedLatent, initSigma);
            packedLatent.Dispose();
            packedLatent = scaled;
        }

        // ── 4. Denoising loop ─────────────────────────────────────────
        Logs.Info("Starting Flux denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f; // Convert timestep back to sigma [0,1]

            // Forward pass: velocity prediction
            Tensor velocityPred = _transformer.Forward(
                _backend, packedLatent, t5Embeddings, sigma,
                clipPooled!, guidanceScale, txtSeqLen, hPacked, wPacked);

            // Scheduler step: Euler on packed latent
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, 64);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            SchedulerStepPacked(newLatent, velocityPred, packedLatent, scheduler, i);
            velocityPred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        clipPooled.Dispose();
        t5Embeddings.Dispose();

        // ── 5. Unpack latent: [1, seqLen, 64] → [1, 16, latentH, latentW] ──
        Tensor unpackedLatent = UnpackLatent(packedLatent, latentH, latentW);
        packedLatent.Dispose();

        // ── 6. VAE decode ─────────────────────────────────────────────
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, unpackedLatent);
        unpackedLatent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 7. Convert to RGB bytes ───────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Flux image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Packs a latent tensor from [B, C, H, W] to [B, H/2*W/2, C*4]. Rearranges 2x2 spatial patches into channel dimension.</summary>
    private static Tensor PackLatent(Tensor latent, int h, int w)
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

    /// <summary>Unpacks a latent tensor from [B, H/2*W/2, C*4] back to [B, C, H, W].</summary>
    private static Tensor UnpackLatent(Tensor packed, int h, int w)
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
