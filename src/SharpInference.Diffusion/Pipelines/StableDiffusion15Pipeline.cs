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

/// <summary>Stable Diffusion 1.5 text-to-image and image-to-image pipeline. Orchestrates CLIP text encoder → UNet denoising loop → VAE decode → RGB image output. Provide a <see cref="VaeEncoder"/> on construction to enable img2img via <see cref="GenerateImg2ImgFromTokens"/>.</summary>
public sealed class StableDiffusion15Pipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _textEncoder;
    private readonly UNet _unet;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private int _disposed;

    /// <summary>Creates a new SD1.5 pipeline. Img2img is unavailable (will throw); use the overload accepting a <see cref="VaeEncoder"/> to enable it.</summary>
    public StableDiffusion15Pipeline(IBackend backend, ClipTextEncoder textEncoder, UNet unet, VaeDecoder vaeDecoder)
        : this(backend, textEncoder, unet, vaeDecoder, vaeEncoder: null)
    {
    }

    /// <summary>Creates a new SD1.5 pipeline with both VAE halves loaded. Required for img2img.</summary>
    public StableDiffusion15Pipeline(IBackend backend, ClipTextEncoder textEncoder, UNet unet, VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder)
    {
        _backend = backend;
        _textEncoder = textEncoder;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
    }

    /// <summary>Generates an image from a text prompt. Returns raw RGB bytes [H, W, 3] and the used seed.</summary>
    public (byte[] rgbData, int width, int height, int seed) Generate(TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Encode text prompt
        Logs.Info("Encoding text prompt...");
        Tensor textEmbeddings = EncodePrompt(request.Prompt, request.NegativePrompt);
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. Create initial noise latent [1, 4, latentH, latentW]
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // 3. Set up scheduler
        IScheduler scheduler = CreateScheduler(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // Scale initial noise by scheduler's initial sigma
        float initSigma = scheduler.InitialNoiseSigma;
        if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
        {
            Tensor scaled = new Tensor(latentShape, DType.F32);
            _backend.Scale(scaled, latent, initSigma);
            latent.Dispose();
            latent = scaled;
        }

        // 4. Denoising loop
        Logs.Info("Starting denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = 0; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // Scale model input (Euler: 1/sqrt(sigma^2+1), others: 1.0)
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
                // Classifier-free guidance: run UNet twice (unconditional + conditional)
                noisePred = ClassifierFreeGuidanceStep(_backend, scaledLatent, t, textEmbeddings, cfgScale);
            }
            else
            {
                // No CFG: single UNet pass with conditional embeddings
                int ctxLen = (int)textEmbeddings.Shape[1] / 2;
                Tensor condEmbeddings = SliceCondEmbeddings(textEmbeddings, ctxLen);
                noisePred = _unet.Forward(_backend, scaledLatent, t, condEmbeddings);
                condEmbeddings.Dispose();
            }

            if (scaledLatent != latent) scaledLatent.Dispose();

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

        textEmbeddings.Dispose();

        // 5. VAE decode
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 6. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Encodes prompt and negative prompt, concatenates for CFG: [2, seqLen, hiddenSize] where [0] is negative, [1] is positive.</summary>
    private Tensor EncodePrompt(string prompt, string negativePrompt)
    {
        // We need the tokenizer externally — for now, expect token IDs to be prepared
        // This method accepts raw text and uses a simple approach for now
        // In production, the tokenizer would be injected
        // For the pipeline, we assume token IDs are generated outside and passed via a different overload

        // For now, we'll create a batch of 2 (negative + positive) using a simplified encoding
        // The actual pipeline would tokenize and then call _textEncoder.Encode
        throw new NotImplementedException(
            "Text encoding requires tokenized input. Use GenerateFromTokens() instead, " +
            "or provide a ClipTokenizer instance to the pipeline.");
    }

    /// <summary>Generates an image from pre-tokenized input. This is the primary entry point for CPU inference testing.</summary>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIds,
        int[] negativePromptTokenIds,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int latentH = request.Height / 8;
        int latentW = request.Width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;

        Logs.Info($"Generating {request.Width}x{request.Height} image, {steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Encode text: batch of [negative, positive]
        Logs.Info("Encoding text prompt...");
        int[][] batchTokenIds = [negativePromptTokenIds, promptTokenIds];
        Tensor textEmbeddings = _textEncoder.Encode(_backend, batchTokenIds);
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. Create initial noise latent [1, 4, latentH, latentW]
        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, seed);

        // 3. Set up scheduler
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

        // 4. Denoising loop
        latent = RunDenoiseLoop(latent, latentShape, textEmbeddings, scheduler, startStep: 0, totalSteps: steps, cfgScale, onProgress);

        textEmbeddings.Dispose();

        // 5. VAE decode
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 6. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Image generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, request.Width, request.Height, seed);
    }

    /// <summary>Generates an image by transforming an existing one. Encodes <see cref="ImageToImageRequest.SourceImage"/> via the VAE encoder, injects noise at the timestep selected by <see cref="ImageToImageRequest.Strength"/>, and runs the denoising loop from there.</summary>
    /// <remarks>Requires a pipeline constructed with a <see cref="VaeEncoder"/>; otherwise throws.</remarks>
    public (byte[] rgbData, int width, int height, int seed) GenerateImg2ImgFromTokens(
        int[] promptTokenIds,
        int[] negativePromptTokenIds,
        ImageToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        if (_vaeEncoder is null)
            throw new InvalidOperationException("Img2img requires a VaeEncoder. Construct the pipeline with the overload that accepts one.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        float strength = Math.Clamp(request.Strength, 0f, 1f);

        // Validate source image shape matches the request resolution. The VAE encoder downsamples
        // by 8× spatially, so source must already be at the target output resolution.
        Tensor source = request.SourceImage;
        if (source.Shape.Rank != 4 || source.Shape[0] != 1 || source.Shape[1] != 3 ||
            source.Shape[2] != height || source.Shape[3] != width)
        {
            throw new ArgumentException(
                $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {source.Shape}.",
                nameof(request));
        }

        Logs.Info($"Img2img {width}x{height}, {steps} steps, strength={strength:F2}, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // 1. Determine starting step from strength. If strength==0, no denoising runs and the source
        // passes through byte-identical. Short-circuit before text encoding / VAE work to avoid waste
        // and to allow callers to skip tokenization entirely (empty token arrays are valid here).
        int initTimesteps = (int)MathF.Round(steps * strength);
        int startStep = Math.Max(steps - initTimesteps, 0);

        if (initTimesteps == 0)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            byte[] passthroughBytes = ImagePostProcessor.TensorToRgbBytes(source);
            return (passthroughBytes, width, height, seed);
        }

        // 2. Encode text
        Logs.Info("Encoding text prompt...");
        int[][] batchTokenIds = [negativePromptTokenIds, promptTokenIds];
        Tensor textEmbeddings = _textEncoder.Encode(_backend, batchTokenIds);
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 3. Encode source image to latent space
        Stopwatch vaeEncSw = Stopwatch.StartNew();
        Tensor sourceLatent = _vaeEncoder.Encode(_backend, source);
        vaeEncSw.Stop();
        Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        // 4. Generate fresh noise (same shape as the encoded latent)
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);

        // 5. Set up scheduler
        IScheduler scheduler = CreateScheduler(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 6. Inject noise at timestep[startStep]: noisy = sourceLatent + noise * sigma[startStep]
        Tensor latent = new Tensor(latentShape, DType.F32);
        scheduler.AddNoise(latent, sourceLatent, noise, startStep);
        sourceLatent.Dispose();
        noise.Dispose();

        Logs.Info($"Starting denoise at step {startStep}/{steps} (running {initTimesteps} steps)");

        // 6. Denoising loop from startStep onward
        latent = RunDenoiseLoop(latent, latentShape, textEmbeddings, scheduler, startStep, totalSteps: steps, cfgScale, onProgress);

        textEmbeddings.Dispose();

        // 7. VAE decode
        Logs.Info("Decoding latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();
        Tensor image = _vaeDecoder.Decode(_backend, latent);
        latent.Dispose();
        vaeSw.Stop();
        Logs.Info($"VAE decode done in {vaeSw.ElapsedMilliseconds}ms");

        // 8. Convert to RGB bytes
        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"Img2img complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>Runs the diffusion denoising loop. Iterates <c>i</c> from <paramref name="startStep"/> through <paramref name="totalSteps"/>-1, applying scheduler input scaling, the UNet (with optional CFG), and one scheduler step per iteration. Returns the final denoised latent. Disposes intermediate latents along the way.</summary>
    private Tensor RunDenoiseLoop(
        Tensor latent,
        TensorShape latentShape,
        Tensor textEmbeddings,
        IScheduler scheduler,
        int startStep,
        int totalSteps,
        float cfgScale,
        Action<GenerationProgress>? onProgress)
    {
        Logs.Info("Starting denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < totalSteps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float t = timesteps[i];

            // Scale model input (Euler: 1/sqrt(sigma^2+1), others: 1.0)
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
                noisePred = ClassifierFreeGuidanceStep(_backend, scaledLatent, t, textEmbeddings, cfgScale);
            }
            else
            {
                int seqLen = (int)textEmbeddings.Shape[1];
                int hiddenSize = (int)textEmbeddings.Shape[2];
                Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
                noisePred = _unet.Forward(_backend, scaledLatent, t, condEmb);
                condEmb.Dispose();
            }

            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePred, latent, i);
            noisePred.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{totalSteps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, totalSteps, stepSw.Elapsed.TotalMilliseconds));
        }

        return latent;
    }

    /// <summary>Runs classifier-free guidance: noise_pred = uncond + cfg_scale * (cond - uncond).</summary>
    private unsafe Tensor ClassifierFreeGuidanceStep(IBackend backend, Tensor latent, float timestep, Tensor textEmbeddings, float cfgScale)
    {
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];

        // Split text embeddings: [0] = negative (uncond), [1] = positive (cond)
        Tensor uncondEmb = SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);

        // Run UNet twice
        Tensor uncondNoise = _unet.Forward(backend, latent, timestep, uncondEmb);
        Tensor condNoise = _unet.Forward(backend, latent, timestep, condEmb);
        uncondEmb.Dispose();
        condEmb.Dispose();

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

    /// <summary>Extracts a single element from the batch dimension of a [B, seqLen, hiddenSize] tensor.</summary>
    private static unsafe Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
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

    /// <summary>Slices conditional embeddings from concatenated [uncond, cond] tensor.</summary>
    private static unsafe Tensor SliceCondEmbeddings(Tensor embeddings, int ctxLen)
    {
        int hiddenSize = (int)embeddings.Shape[2];
        TensorShape shape = new TensorShape(1, ctxLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);

        float* srcPtr = (float*)embeddings.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int offset = ctxLen * hiddenSize; // Skip uncond
        int elements = ctxLen * hiddenSize;

        for (int i = 0; i < elements; i++)
        {
            dstPtr[i] = srcPtr[offset + i];
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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    /// <summary>Disposes the pipeline. Does not dispose the backend (shared resource).</summary>
    public void Dispose()
    {
        Volatile.Write(ref _disposed, 1);
    }
}
