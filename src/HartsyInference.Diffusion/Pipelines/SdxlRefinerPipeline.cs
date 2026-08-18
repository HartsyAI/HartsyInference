using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>SDXL refiner pipeline. Refines an existing image using the SDXL refiner UNet (4-level, CLIP-G-only conditioning, aesthetic-score ADM). Cross-model refining works for any base pipeline (SD1.5/SDXL/Flux/Z-Image) because the handoff is in pixel space — strength controls how aggressively to polish (0.3 typical, 0.0 pass-through). Same-VAE latent handoff is a deferred SDXL→SDXL optimization.</summary>
public sealed class SdxlRefinerPipeline : DiffusionPipelineBase
{
    private readonly ClipTextEncoder _clipG;
    private readonly UNet _refinerUnet;
    private readonly VaeEncoder _vaeEncoder;
    private readonly VaeDecoder _vaeDecoder;
    private readonly float _vaeScalingFactor;

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
        : base(backend)
    {
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
        // Wrap-pad every conv backend for this call so the output tiles seamlessly; restores on dispose. The refiner
        // stage has to match the base stage or the refine pass would re-introduce the seam the base avoided.
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfgScale, int width, int height) = GenerationDefaults.Sdxl.Resolve(request);
        int latentH = height / 8;
        int latentW = width / 8;
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
        (Tensor textEmbeddings, Tensor? pooledOutput) = _clipG.EncodePenultimate(Backend, batchTokenIdsG, eosPositions, request.ClipSkip ?? 2);
        if (pooledOutput is null)
            throw new InvalidOperationException("CLIP-G must produce a pooled output for SDXL refiner.");
        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // 2. ADM scalars: refiner uses 5 values per branch (orig_h/w, crop_top/left, aesthetic_score).
        //    The aesthetic_score differs between cond/uncond branches, so we build two arrays.
        float[] sizeConditionPos =
        [
            height, width,
            0f, 0f,
            request.AestheticScore,
        ];
        float[] sizeConditionNeg =
        [
            height, width,
            0f, 0f,
            request.NegativeAestheticScore,
        ];

        // 3. Encode source image via the VAE encoder
        Stopwatch vaeEncSw = Stopwatch.StartNew();
        Tensor sourceLatent = _vaeEncoder.Encode(Backend, source);
        vaeEncSw.Stop();
        Logs.Info($"VAE encode done in {vaeEncSw.ElapsedMilliseconds}ms");

        TensorShape latentShape = new TensorShape(1, 4, latentH, latentW);

        // 4. Generate fresh noise + scheduler setup
        Tensor noise = SeedGenerator.CreateNoise(latentShape, seed);
        IScheduler scheduler = SchedulerFactory.Create(request.Scheduler);
        scheduler.SetTimesteps(steps);

        // 5. Inject noise at sigma[startStep]
        Tensor latent = new Tensor(latentShape, DType.F32);
        scheduler.AddNoise(latent, sourceLatent, noise, startStep);
        sourceLatent.Dispose();
        noise.Dispose();

        Logs.Info($"Refiner denoise from step {startStep}/{steps} ({initTimesteps} steps)");

        // 6. Refiner denoise loop
        // Bulk-upload refiner UNet weights before the loop. Paired with FreeWeights below
        // the VAE handoff. No-op on backends without a weight cache.
        Backend.PreloadWeights(_refinerUnet.EnumerateWeights());
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
                Backend.Scale(scaledLatent, latent, inputScale);
            }
            else
            {
                scaledLatent = latent;
            }

            Tensor unetInput = useF16
                ? DtypeCastHelper.EnsureDtype(Backend, scaledLatent, DType.F16, disposeSourceOnCast: false)
                : scaledLatent;

            Tensor noisePred;
            if (cfgScale > 1.0f)
            {
                noisePred = ClassifierFreeGuidanceStep(unetInput, t, textEmbeddings, pooledOutput, sizeConditionPos, sizeConditionNeg, cfgScale);
            }
            else
            {
                int seqLen = (int)textEmbeddings.Shape[1];
                int hiddenSize = (int)textEmbeddings.Shape[2];
                Tensor condEmb = CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
                int pooledDim = (int)pooledOutput.Shape[1];
                Tensor condPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDim);
                noisePred = _refinerUnet.Forward(Backend, unetInput, t, condEmb, condPooled, sizeConditionPos);
                condEmb.Dispose();
                condPooled.Dispose();
            }

            if (unetInput != scaledLatent) unetInput.Dispose();
            if (scaledLatent != latent) scaledLatent.Dispose();

            Tensor noisePredF32 = DtypeCastHelper.EnsureF32(Backend, noisePred);

            Tensor newLatent = new Tensor(latentShape, DType.F32);
            scheduler.Step(newLatent, noisePredF32, latent, i);
            noisePredF32.Dispose();
            latent.Dispose();
            latent = newLatent;

            stepSw.Stop();
            Logs.Info($"Refiner step {i + 1}/{steps} (t={t:F1}) done in {stepSw.ElapsedMilliseconds}ms");
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = latent,
                LatentArch = LatentArchitecture.Sdxl,
            });
        }

        textEmbeddings.Dispose();
        pooledOutput.Dispose();

        // 7. VAE decode
        Backend.Sync();
        Backend.FreeWeights(_refinerUnet.EnumerateWeights());
        Logs.Info("Decoding refined latents to image...");
        Stopwatch vaeSw = Stopwatch.StartNew();

        bool vaeF16 = (_vaeDecoder.EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32) == DType.F16;
        Tensor vaeInput = vaeF16
            ? DtypeCastHelper.EnsureDtype(Backend, latent, DType.F16)
            : latent;

        // Tiled decode: caps im2col workspace at ~2.4 GB per tile.
        Tensor image = _vaeDecoder.DecodeTiled(Backend, vaeInput);
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

        Tensor uncondEmb = CfgHelper.SliceBatchElement(textEmbeddings, 0, seqLen, hiddenSize);
        Tensor condEmb = CfgHelper.SliceBatchElement(textEmbeddings, 1, seqLen, hiddenSize);
        Tensor uncondPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 0, pooledDim);
        Tensor condPooled = CfgHelper.SliceBatchElement1D(pooledOutput, 1, pooledDim);

        // Note: refiner uses different aesthetic scalars per branch
        Tensor uncondNoise = _refinerUnet.Forward(Backend, latent, timestep, uncondEmb, uncondPooled, sizeConditionNeg);
        Tensor condNoise = _refinerUnet.Forward(Backend, latent, timestep, condEmb, condPooled, sizeConditionPos);
        uncondEmb.Dispose();
        condEmb.Dispose();
        uncondPooled.Dispose();
        condPooled.Dispose();

        Tensor uncondF32 = DtypeCastHelper.EnsureF32(Backend, uncondNoise);
        Tensor condF32 = DtypeCastHelper.EnsureF32(Backend, condNoise);
        Tensor output = CfgHelper.ApplyCfg(uncondF32, condF32, cfgScale);
        uncondF32.Dispose();
        condF32.Dispose();
        return output;
    }
}
