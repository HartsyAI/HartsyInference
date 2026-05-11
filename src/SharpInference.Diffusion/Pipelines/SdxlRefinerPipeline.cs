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

/// <summary>SDXL refiner pipeline. Refines an existing image using the SDXL refiner UNet (4-level, CLIP-G-only conditioning, aesthetic-score ADM). Cross-model refining works for any base pipeline (SD1.5/SDXL/Flux/Z-Image) because the handoff is in pixel space — strength controls how aggressively to polish (0.3 typical, 0.0 pass-through). Same-VAE latent handoff is a deferred SDXL→SDXL optimization.</summary>
public sealed unsafe class SdxlRefinerPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _refinerUnet;
    private readonly VaeEncoder _vaeEncoder;
    private readonly VaeDecoder _vaeDecoder;
    private readonly float _vaeScalingFactor;
    private int _disposed;

    /// <summary>Creates a new SDXL refiner pipeline.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="clipG">OpenCLIP ViT-bigG/14 text encoder (the refiner's only text encoder).</param>
    /// <param name="refinerUnet">SDXL refiner UNet, configured with <see cref="UNetConfig.SdxlRefiner"/>.</param>
    /// <param name="vaeEncoder">VAE encoder (configured with <see cref="VaeConfig.Sdxl"/>) — required for img2img-style refining.</param>
    /// <param name="vaeDecoder">VAE decoder (configured with <see cref="VaeConfig.Sdxl"/>).</param>
    /// <param name="vaeScalingFactor">VAE scaling factor. Default 0.13025 matches the SDXL VAE.</param>
    public SdxlRefinerPipeline(
        IBackend backend,
        ClipTextEncoder clipG,
        UNet refinerUnet,
        VaeEncoder vaeEncoder,
        VaeDecoder vaeDecoder,
        float vaeScalingFactor = 0.13025f)
    {
        _backend = backend;
        _clipG = clipG;
        _refinerUnet = refinerUnet;
        _vaeEncoder = vaeEncoder;
        _vaeDecoder = vaeDecoder;
        _vaeScalingFactor = vaeScalingFactor;
    }

    /// <summary>Refines a source image. Encodes the source via the VAE encoder, injects noise at the timestep
    /// selected by <see cref="ImageToImageRequest.Strength"/>, and runs the refiner UNet from there.</summary>
    /// <param name="promptTokenIdsG">Prompt token IDs for CLIP-G [seqLen]. Tokenize using the SDXL CLIP-G tokenizer.</param>
    /// <param name="negativePromptTokenIdsG">Negative prompt token IDs for CLIP-G [seqLen].</param>
    /// <param name="promptEosPositionG">Position of EOS token in prompt for CLIP-G (used to extract pooled output).</param>
    /// <param name="negativeEosPositionG">Position of EOS token in negative prompt.</param>
    /// <param name="request">Refiner request with source image, strength, and aesthetic scores.</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) RefineFromTokens(
        int[] promptTokenIdsG,
        int[] negativePromptTokenIdsG,
        int promptEosPositionG,
        int negativeEosPositionG,
        SdxlRefinerRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width;
        int height = request.Height;
        int latentH = height / 8;
        int latentW = width / 8;
        int steps = request.Steps;
        float cfgScale = request.CfgScale;
        float strength = Math.Clamp(request.Strength, 0f, 1f);

        Tensor source = request.SourceImage;
        if (source.Shape.Rank != 4 || source.Shape[0] != 1 || source.Shape[1] != 3 ||
            source.Shape[2] != height || source.Shape[3] != width)
        {
            throw new ArgumentException(
                $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {source.Shape}.",
                nameof(request));
        }

        Logs.Info($"SDXL Refiner: {width}x{height}, {steps} steps, strength={strength:F2}, cfg={cfgScale}, " +
                  $"aesthetic={request.AestheticScore:F1}/{request.NegativeAestheticScore:F1}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // Strength=0 → byte-identical pass-through.
        int initTimesteps = (int)MathF.Round(steps * strength);
        int startStep = Math.Max(steps - initTimesteps, 0);
        if (initTimesteps == 0)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(source), width, height, seed);
        }

        // 1. CLIP-G text encoding (refiner uses only CLIP-G, not CLIP-L)
        Logs.Info("Encoding text with CLIP-G...");
        int[][] batchTokenIdsG = [negativePromptTokenIdsG, promptTokenIdsG];
        int[] eosPositions = [negativeEosPositionG, promptEosPositionG];
        (Tensor textEmbeddings, Tensor? pooledOutput) = _clipG.EncodePenultimate(_backend, batchTokenIdsG, eosPositions);
        if (pooledOutput is null)
            throw new InvalidOperationException("CLIP-G must produce a pooled output for SDXL refiner.");
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. ADM scalars: refiner uses 5 values per branch (orig_h/w, crop_top/left, aesthetic_score).
        //    The aesthetic_score differs between cond/uncond branches, so we build two arrays.
        float[] sizeConditionPos =
        [
            request.Height, request.Width,
            0f, 0f,
            request.AestheticScore,
        ];
        float[] sizeConditionNeg =
        [
            request.Height, request.Width,
            0f, 0f,
            request.NegativeAestheticScore,
        ];

        // 3. Encode source image via the VAE encoder
        Stopwatch vaeEncSw = Stopwatch.StartNew();
        Tensor sourceLatent = _vaeEncoder.Encode(_backend, source);
        vaeEncSw.Stop();
        Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        // 4. Generate fresh noise + scheduler setup
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
        IScheduler scheduler = CreateScheduler(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 5. Inject noise at sigma[startStep]
        Tensor latent = new Tensor(latentShape, DType.F32);
        scheduler.AddNoise(latent, sourceLatent, noise, startStep);
        sourceLatent.Dispose();
        noise.Dispose();

        Logs.Info($"Refiner denoise from step {startStep}/{steps} ({initTimesteps} steps)");

        // 6. Refiner denoise loop
        bool useF16 = (_refinerUnet.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        for (int i = startStep; i < steps; i++)
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
                noisePred = ClassifierFreeGuidanceStep(unetInput, t, textEmbeddings, pooledOutput, sizeConditionPos, sizeConditionNeg, cfgScale);
            }
            else
            {
                int seqLen = (int)textEmbeddings.Shape[1];
                int hiddenSize = (int)textEmbeddings.Shape[2];
                Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput.Shape[1];
                Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = _refinerUnet.Forward(_backend, unetInput, t, condEmb, condPooled, sizeConditionPos);
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
            Logs.Info($"Refiner step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds));
        }

        textEmbeddings.Dispose();
        pooledOutput.Dispose();

        // 7. VAE decode
        _backend.Sync();
        _backend.FreeWeights(_refinerUnet.EnumerateWeights());
        Logs.Info("Decoding refined latents to image...");
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

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(image);
        image.Dispose();

        sw.Stop();
        Logs.Info($"SDXL refiner complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    /// <summary>CFG for the refiner: separate ADM size_conditions for cond/uncond (because aesthetic_score differs).</summary>
    private Tensor ClassifierFreeGuidanceStep(
        Tensor latent, float timestep,
        Tensor textEmbeddings, Tensor pooledOutput,
        float[] sizeConditionPos, float[] sizeConditionNeg, float cfgScale)
    {
        int seqLen = (int)textEmbeddings.Shape[1];
        int hiddenSize = (int)textEmbeddings.Shape[2];
        int pooledDim = (int)pooledOutput.Shape[1];

        Tensor uncondEmb = SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor uncondPooled = SliceBatchElement1D(pooledOutput, 0, pooledDim);
        Tensor condPooled = SliceBatchElement1D(pooledOutput, 1, pooledDim);

        // Note: refiner uses different aesthetic scalars per branch
        Tensor uncondNoise = _refinerUnet.Forward(_backend, latent, timestep, uncondEmb, uncondPooled, sizeConditionNeg);
        Tensor condNoise = _refinerUnet.Forward(_backend, latent, timestep, condEmb, condPooled, sizeConditionPos);
        uncondEmb.Dispose();
        condEmb.Dispose();
        uncondPooled.Dispose();
        condPooled.Dispose();

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

    private static Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;
        for (int i = 0; i < elements; i++) dstPtr[i] = srcPtr[srcOffset + i];
        return slice;
    }

    private static Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;
        for (int i = 0; i < dim; i++) dstPtr[i] = srcPtr[srcOffset + i];
        return slice;
    }

    private static IScheduler CreateScheduler(string? name) => (name?.ToLowerInvariant()) switch
    {
        "ddim" => new DdimScheduler(),
        "dpm++2m" or "dpmpp2m" => new DpmPlusPlus2MScheduler(),
        _ => new EulerDiscreteScheduler(),
    };

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Disposes the pipeline. Does not dispose the backend or model components (shared resources).</summary>
    public void Dispose() => Volatile.Write(ref _disposed, 1);
}
