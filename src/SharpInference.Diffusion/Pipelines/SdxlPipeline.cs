using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Schedulers;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>SDXL text-to-image pipeline. Orchestrates dual CLIP text encoding (CLIP-L + CLIP-G) → UNet denoising with ADM conditioning → VAE decode → RGB image output.</summary>
public sealed unsafe class SdxlPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly float _vaeScalingFactor;
    private int _disposed;

    /// <summary>Creates a new SDXL pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipL">CLIP ViT-L/14 text encoder (text_encoder).</param>
    /// <param name="clipG">OpenCLIP ViT-bigG/14 text encoder (text_encoder_2).</param>
    /// <param name="unet">SDXL UNet (configured with UNetConfig.SdxlBase).</param>
    /// <param name="vaeDecoder">VAE decoder (configured with VaeConfig.Sdxl, scaling factor 0.13025).</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Default: 0.13025 for SDXL.</param>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, float vaeScalingFactor = 0.13025f)
    {
        _backend = backend;
        _clipL = clipL;
        _clipG = clipG;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _vaeScalingFactor = vaeScalingFactor;
    }

    /// <summary>Generates an image from pre-tokenized input for both CLIP encoders.</summary>
    /// <param name="promptTokenIdsL">Prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="negativePromptTokenIdsL">Negative prompt token IDs for CLIP-L [seqLen].</param>
    /// <param name="promptTokenIdsG">Prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="negativePromptTokenIdsG">Negative prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="promptEosPositionG">Position of EOS token in prompt for CLIP-G (for pooled output).</param>
    /// <param name="negativeEosPositionG">Position of EOS token in negative prompt for CLIP-G.</param>
    /// <param name="request">Generation parameters.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsL,
        int[] negativePromptTokenIdsL,
        int[] promptTokenIdsG,
        int[] negativePromptTokenIdsG,
        int promptEosPositionG,
        int negativeEosPositionG,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"SDXL: Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Encode text with dual CLIP encoders
        Logs.Info("Encoding text with dual CLIP encoders...");

        // CLIP-L: penultimate hidden states [2, 77, 768] (batch: [negative, positive])
        int[][] batchTokenIdsL = [negativePromptTokenIdsL, promptTokenIdsL];
        (Tensor clipLHidden, _) = _clipL.EncodePenultimate(_backend, batchTokenIdsL, [0, 0]);

        // CLIP-G: penultimate hidden states [2, 77, 1280] + pooled output [2, 1280]
        int[][] batchTokenIdsG = [negativePromptTokenIdsG, promptTokenIdsG];
        int[] eosPositions = [negativeEosPositionG, promptEosPositionG];
        (Tensor clipGHidden, Tensor? pooledOutput) = _clipG.EncodePenultimate(_backend, batchTokenIdsG, eosPositions);

        // Concatenate hidden states along feature dimension: [2, 77, 768] + [2, 77, 1280] = [2, 77, 2048]
        Tensor textEmbeddings = ConcatAlongLastDim(clipLHidden, clipGHidden);
        clipLHidden.Dispose();
        clipGHidden.Dispose();

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. Build ADM conditioning scalars for SDXL base
        // Default: original size = target size = requested size, no crop
        float[] sizeCondition =
        [
            request.Height, request.Width,   // orig_height, orig_width
            0f, 0f,                          // crop_top, crop_left
            request.Height, request.Width    // target_height, target_width
        ];

        // 3. Create initial noise latent [1, 4, latentH, latentW]
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // 4. Set up scheduler
        IScheduler scheduler = CreateScheduler(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // Scale initial noise
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // 5. Denoising loop
        Logs.Info("Starting SDXL denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // Scale model input
            float inputScale = scheduler.ScaleModelInput(i);
            Tensor scaledLatent;
            if (MathF.Abs(inputScale - 1.0f) > 1e-6f)
            {
                scaledLatent = new Tensor(latentShape, DType.F32);
                _backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }

            Tensor noisePred;
            if (cfgScale > 1.0f)
            {
                noisePred = ClassifierFreeGuidanceStep(scaledLatent, t, textEmbeddings, pooledOutput!, sizeCondition, cfgScale);
            }
            else
            {
                // Single pass with conditional embeddings only
                int seqLen = (int)textEmbeddings.Shape[1];
                int hiddenSize = (int)textEmbeddings.Shape[2];
                Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput!.Shape[1];
                Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = _unet.Forward(_backend, scaledLatent, t, condEmb, condPooled, sizeCondition);
                condEmb.Dispose();
                condPooled.Dispose();
            }

            if (scaledLatent != latent) scaledLatent.Dispose();

            // Scheduler step
            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            string cacheInfo = GetBackendCacheStats();
            Logs.Info($"Step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms{cacheInfo}");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        textEmbeddings.Dispose();
        pooledOutput?.Dispose();

        // 6. VAE decode — free UNet weights to reclaim VRAM for high-res VAE conv2d buffers
        _backend.Sync();
        _backend.FreeWeights(_unet.EnumerateWeights());
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 7. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SDXL image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Runs classifier-free guidance for SDXL: noise_pred = uncond + cfg_scale * (cond - uncond).</summary>
    private Tensor ClassifierFreeGuidanceStep(Tensor latent, float timestep, Tensor textEmbeddings, Tensor pooledOutput, float[] sizeCondition, float cfgScale)
    {
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];
        int pooledDim = (int)pooledOutput.Shape[1];

        // Split text embeddings: [0] = negative (uncond), [1] = positive (cond)
        Tensor uncondEmb = SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor uncondPooled = SliceBatchElement1D(pooledOutput, 0, pooledDim);
        Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);

        // Run UNet twice with ADM conditioning
        Tensor uncondNoise = _unet.Forward(_backend, latent, timestep, uncondEmb, uncondPooled, sizeCondition);
        Tensor condNoise = _unet.Forward(_backend, latent, timestep, condEmb, condPooled, sizeCondition);
        uncondEmb.Dispose();
        condEmb.Dispose();
        uncondPooled.Dispose();
        condPooled.Dispose();

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

                for (int d = 0; d < dimA; d++)
                {
                    outPtr[outOffset + d] = aPtr[aOffset + d];
                }
                for (int d = 0; d < dimB; d++)
                {
                    outPtr[outOffset + dimA + d] = bPtr[bOffset + d];
                }
            }
        }

        return output;
    }

    /// <summary>Extracts a single element from the batch dimension of a [B, seqLen, hiddenSize] tensor.</summary>
    private static Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;

        for (int i = 0; i < elements; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }

        return slice;
    }

    /// <summary>Extracts a single element from the batch dimension of a [B, dim] tensor → [1, dim].</summary>
    private static Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;

        for (int i = 0; i < dim; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }

        return slice;
    }

    /// <summary>Creates the requested scheduler.</summary>
    private static IScheduler CreateScheduler(string? name)
    {
        return (name?.ToLowerInvariant()) switch
        {
            "ddim" => new DdimScheduler(),
            "dpm++2m" or "dpmpp2m" => new DpmPlusPlus2MScheduler(),
            _ => new EulerDiscreteScheduler(),
        };
    }

    /// <summary>Gets GPU cache stats string if the backend supports it.</summary>
    private string GetBackendCacheStats()
    {
        System.Reflection.MethodInfo? method = _backend.GetType().GetMethod("GetGpuCacheStats");
        if (method != null)
        {
            (long cachedBytes, long hits, long misses) stats = ((long, long, long))method.Invoke(_backend, null)!;
            return $" (GPU cache: {stats.cachedBytes / 1024 / 1024}MB, hits={stats.hits}, misses={stats.misses})";
        }
        return "";
    }

    /// <summary>Evicts GPU weight cache if the backend supports it (CudaBackend).</summary>
    private void EvictBackendCache(string stage)
    {
        System.Reflection.MethodInfo? method = _backend.GetType().GetMethod("EvictGpuCache");
        method?.Invoke(_backend, null);
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
