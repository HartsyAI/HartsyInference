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
        LogTensorStats("CLIP hidden (full)", clipLHidden);
        Tensor clipPooled = ExtractEosHiddenState(clipLHidden, promptEosPositionL);
        clipLHidden.Dispose();

        // T5-XXL: per-token embeddings [B, seqLen, 4096]
        int[][] batchTokenIdsT5 = [promptTokenIdsT5];
        int[][]? batchMaskT5 = promptAttentionMaskT5 is not null ? [promptAttentionMaskT5] : null;
        Tensor t5Embeddings = _t5.Encode(_backend, batchTokenIdsT5, batchMaskT5);

        int txtSeqLen = (int)t5Embeddings.Shape[1];
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms (T5 seqLen={txtSeqLen})");
        LogTensorStats("CLIP pooled", clipPooled);
        LogTensorStats("T5 embeddings", t5Embeddings);

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

            LogTensorStats($"Step {i+1} velocity", velocityPred);
            LogPerLatentChannelMeanPacked($"Step {i+1} velocity", velocityPred);

            // Scheduler step: Euler on packed latent
            TensorShape packedStepShape = new TensorShape(1, imgSeqLen, 64);
            Tensor newLatent = new Tensor(packedStepShape, DType.F32);
            SchedulerStepPacked(newLatent, velocityPred, packedLatent, scheduler, i);
            velocityPred.Dispose();
            packedLatent.Dispose();
            packedLatent = newLatent;

            LogPerLatentChannelMeanPacked($"Step {i+1} latent", packedLatent);

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        clipPooled.Dispose();
        t5Embeddings.Dispose();

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
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, unpackedLatent);
        unpackedLatent.Dispose();
        LogTensorStats("VAE output", image);
        LogPerChannelStats("VAE output", image);
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // ── 7. Convert to RGB bytes ───────────────────────────────────
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Flux image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    private static void LogTensorStats(string name, Tensor tensor)
    {
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
