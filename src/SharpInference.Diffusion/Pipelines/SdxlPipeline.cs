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

/// <summary>SDXL text-to-image and image-to-image pipeline. Orchestrates dual CLIP text encoding (CLIP-L + CLIP-G) → UNet denoising with ADM conditioning → VAE decode → RGB image output.
/// <para>Img2img is selected by passing an <see cref="ImageToImageRequest"/> (instead of <see cref="TextToImageRequest"/>) to <see cref="GenerateFromTokens"/> — same method, single API. Requires a <see cref="VaeEncoder"/> on construction. This also unlocks the cross-model refining pattern: any base pipeline's RGB output can be fed through this pipeline as a refiner via <see cref="ImagePostProcessor.RgbBytesToTensor"/> + img2img with low <c>Strength</c>.</para>
/// </summary>
public sealed unsafe class SdxlPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly float _vaeScalingFactor;
    private int _disposed;

    /// <summary>Creates a new SDXL pipeline. Img2img is unavailable; use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipL">CLIP ViT-L/14 text encoder (text_encoder).</param>
    /// <param name="clipG">OpenCLIP ViT-bigG/14 text encoder (text_encoder_2).</param>
    /// <param name="unet">SDXL UNet (configured with UNetConfig.SdxlBase).</param>
    /// <param name="vaeDecoder">VAE decoder (configured with VaeConfig.Sdxl, scaling factor 0.13025).</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Default: 0.13025 for SDXL.</param>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, float vaeScalingFactor = 0.13025f)
        : this(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder: null, vaeScalingFactor)
    {
    }

    /// <summary>Creates a new SDXL pipeline with both VAE halves loaded. Required for img2img and for use as a cross-model refiner.</summary>
    public SdxlPipeline(IBackend backend, ClipTextEncoder clipL, ClipTextEncoder clipG, UNet unet, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, float vaeScalingFactor = 0.13025f)
    {
        _backend = backend;
        _clipL = clipL;
        _clipG = clipG;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _vaeScalingFactor = vaeScalingFactor;
    }

    /// <summary>Generates an image from pre-tokenized dual-CLIP input. Handles both text-to-image and image-to-image via the runtime type of <paramref name="request"/>:
    /// <list type="bullet">
    /// <item>Plain <see cref="TextToImageRequest"/> → text-to-image (initial latent = noise * initSigma, denoise from step 0).</item>
    /// <item><see cref="ImageToImageRequest"/> → image-to-image (initial latent = VAE-encoded source + noise * sigma[startStep], denoise from <c>startStep = steps - round(steps * Strength)</c>). Requires a <see cref="VaeEncoder"/>.</item>
    /// </list>
    /// The two paths share the entire dual-CLIP encoding, ADM conditioning, denoise loop, and VAE decode pipeline. Strength=0 short-circuits to byte-identical pass-through.
    /// </summary>
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
        bool isImg2Img = request is ImageToImageRequest;
        if (isImg2Img && _vaeEncoder is null)
            throw new InvalidOperationException("ImageToImageRequest requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        // Img2img: validate source shape + handle strength=0 short-circuit BEFORE any model work.
        int startStep = 0;
        if (request is ImageToImageRequest img2img)
        {
            Tensor src = img2img.SourceImage;
            if (src.Shape.Rank != 4 || src.Shape[0] != 1 || src.Shape[1] != 3 ||
                src.Shape[2] != height || src.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {src.Shape}.",
                    nameof(request));
            }

            float strength = Math.Clamp(img2img.Strength, 0f, 1f);
            int initTimesteps = (int)MathF.Round(steps * strength);
            startStep = Math.Max(steps - initTimesteps, 0);

            if (initTimesteps == 0)
            {
                Logs.Info("Strength=0; passing source through unchanged");
                return (ImagePostProcessor.TensorToRgbBytes(src), width, height, seed);
            }
        }

        string mode = isImg2Img ? $"img2img (start={startStep}/{steps})" : "txt2img";
        Logs.Info($"SDXL {mode}: {width}x{height}, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Dual CLIP text encoding (shared by both paths)
        Logs.Info("Encoding text with dual CLIP encoders...");
        int[][] batchTokenIdsL = [negativePromptTokenIdsL, promptTokenIdsL];
        (Tensor clipLHidden, _) = _clipL.EncodePenultimate(_backend, batchTokenIdsL, [0, 0]);

        int[][] batchTokenIdsG = [negativePromptTokenIdsG, promptTokenIdsG];
        int[] eosPositions = [negativeEosPositionG, promptEosPositionG];
        (Tensor clipGHidden, Tensor? pooledOutput) = _clipG.EncodePenultimate(_backend, batchTokenIdsG, eosPositions);

        Tensor textEmbeddings = ConcatAlongLastDim(clipLHidden, clipGHidden);
        clipLHidden.Dispose();
        clipGHidden.Dispose();
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. ADM size conditioning (orig_size = target_size = request resolution; no crop)
        float[] sizeCondition =
        [
            request.Height, request.Width,
            0f, 0f,
            request.Height, request.Width,
        ];

        // 3. Set up scheduler
        IScheduler scheduler = CreateScheduler(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 4. Build initial latent — t2i: noise * initSigma; img2img: vaeEncoder + AddNoise at startStep.
        Tensor latent = BuildInitialLatent(request, scheduler, latentShape, seed, startStep);

        // 5. Denoise loop (both paths run the same loop from their respective startStep)
        bool useF16 = (_unet.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;
        latent = RunDenoiseLoop(latent, latentShape, textEmbeddings, pooledOutput!, sizeCondition, scheduler, useF16, startStep, totalSteps: steps, cfgScale, onProgress);

        textEmbeddings.Dispose();
        pooledOutput?.Dispose();

        // 6. VAE decode — free UNet weights to reclaim VRAM for high-res VAE conv2d buffers
        _backend.Sync();
        _backend.FreeWeights(_unet.EnumerateWeights());
        Logs.Verbose("Decoding latents to image (tiled F32 path)...");
        Stopwatch vaeSw = Stopwatch.StartNew();

        bool vaeF16 = (_vaeDecoder.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;
        Tensor vaeInput = latent;
        if (vaeF16 && latent.DType != DType.F16)
        {
            vaeInput = new Tensor(latent.Shape, DType.F16);
            _backend.CastToF16(vaeInput, latent);
            latent.Dispose();
        }

        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Tensor image = _vaeDecoder.DecodeTiled(_backend, vaeInput);
        vaeInput.Dispose();
        vaeSw.Stop();
        Logs.Verbose($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 7. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SDXL {mode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Builds the starting latent for the denoise loop. T2i: fresh noise scaled by <c>InitialNoiseSigma</c>. Img2img: VAE-encoded source plus fresh noise injected at <c>sigma[startStep]</c> via <c>scheduler.AddNoise</c>.</summary>
    private Tensor BuildInitialLatent(TextToImageRequest request, IScheduler scheduler, TensorShape latentShape, int seed, int startStep)
    {
        if (request is ImageToImageRequest img2img)
        {
            Stopwatch vaeEncSw = Stopwatch.StartNew();
            Tensor sourceLatent = _vaeEncoder!.Encode(_backend, img2img.SourceImage);
            vaeEncSw.Stop();
            Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

            Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
            Tensor latent = new Tensor(latentShape, DType.F32);
            scheduler.AddNoise(latent, sourceLatent, noise, startStep);
            sourceLatent.Dispose();
            noise.Dispose();
            return latent;
        }

        Tensor t2iNoise = SeedGenerator.CreateNoise(latentShape, seed);
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, t2iNoise, initSigma);
            t2iNoise.Dispose();
            return scaled;
        }
        return t2iNoise;
    }

    /// <summary>Runs the SDXL denoising loop from <paramref name="startStep"/> through <paramref name="totalSteps"/>-1. Handles input-scale, F16 cast, dual-conditioning UNet, CFG, and scheduler step. Returns the final denoised F32 latent. Disposes intermediate latents along the way.</summary>
    private Tensor RunDenoiseLoop(
        Tensor latent,
        TensorShape latentShape,
        Tensor textEmbeddings,
        Tensor pooledOutput,
        float[] sizeCondition,
        IScheduler scheduler,
        bool useF16,
        int startStep,
        int totalSteps,
        float cfgScale,
        Action<GenerationProgress>? onProgress)
    {
        Logs.Info("Starting SDXL denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < totalSteps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

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

            Tensor unetInput = scaledLatent;
            if (useF16 && scaledLatent.DType != DType.F16)
            {
                unetInput = new Tensor(scaledLatent.Shape, DType.F16);
                _backend.CastToF16(unetInput, scaledLatent);
            }

            Tensor noisePred;
            if (cfgScale > 1.0f)
            {
                noisePred = ClassifierFreeGuidanceStep(unetInput, t, textEmbeddings, pooledOutput, sizeCondition, cfgScale);
            }
            else
            {
                int seqLen = (int)textEmbeddings.Shape[1];
                int hiddenSize = (int)textEmbeddings.Shape[2];
                Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput.Shape[1];
                Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = _unet.Forward(_backend, unetInput, t, condEmb, condPooled, sizeCondition);
                condEmb.Dispose();
                condPooled.Dispose();
            }

            if (unetInput != scaledLatent) unetInput.Dispose();
            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor noisePredF32 = noisePred;
            if (noisePred.DType == DType.F16)
            {
                noisePredF32 = new Tensor(noisePred.Shape, DType.F32);
                _backend.CastToF32(noisePredF32, noisePred);
                noisePred.Dispose();
            }

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePredF32, latent, i);
            noisePredF32.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            string cacheInfo = GetBackendCacheStats();
            Logs.Info($"Step {i + 1}/{totalSteps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms{cacheInfo}");
            onProgress?.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds));
        }

        return latent;
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
        // UNet output may be F16 — cast to F32 for CPU-side arithmetic, return F32 for scheduler
        Tensor uncondF32 = uncondNoise;
        Tensor condF32 = condNoise;
        if (uncondNoise.DType == DType.F16)
        {
            uncondF32 = new Tensor(uncondNoise.Shape, DType.F32);
            _backend.CastToF32(uncondF32, uncondNoise);
            uncondNoise.Dispose();
            condF32 = new Tensor(condNoise.Shape, DType.F32);
            _backend.CastToF32(condF32, condNoise);
            condNoise.Dispose();
        }

        Tensor output = new Tensor(latent.Shape, DType.F32);
        float* uncPtr = (float*)uncondF32.DataPointer;
        float* conPtr = (float*)condF32.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)latent.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = uncPtr[i] + cfgScale * (conPtr[i] - uncPtr[i]);
        }

        uncondF32.Dispose();
        condF32.Dispose();

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
