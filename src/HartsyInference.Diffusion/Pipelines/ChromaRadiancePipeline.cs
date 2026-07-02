using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Chroma Radiance text-to-image pipeline (<c>lodestones/Chroma1-Radiance</c>) — pixel-space, VAE-free.
/// T5-XXL encode (identical text path to <see cref="ChromaPipeline"/>, including the "first padding token unmasked"
/// mask rule) → <see cref="ChromaRadianceTransformer"/> denoising directly on RGB in [-1, 1] → bytes.
///
/// Sampling differences vs classic Chroma:
/// <list type="bullet">
///   <item><b>x0 prediction</b> — the model predicts the clean image. Each step converts to velocity via
///         <see cref="X0Prediction.ToVelocity"/> (<c>v = (x_t − x0) / max(t, ε)</c>, matching ComfyUI), CFG-combines
///         on v, then takes the flow-match Euler step.</item>
///   <item><b>Static shift 1.0</b> (ComfyUI's Chroma sampling default — no dynamic shift), default 50 steps, CFG 3.5.</item>
///   <item><b>Pixel space</b> — no latent packing, no VAE. Dimensions are padded up to a multiple of the 16-px patch
///         and the output is cropped back. Previews are the in-flight image itself
///         (<see cref="LatentArchitecture.ChromaRadiance"/>).</item>
///   <item><b>Img2img / inpaint without a VAE</b> — the source image IS the clean sample: it's padded to the
///         patch grid and noised directly via flow-matching <c>AddNoise</c> at <c>sigma[startStep]</c>. Masked
///         inpaint blends per step in pixel space (mask used at full resolution — no downsample).</item>
/// </list></summary>
public sealed unsafe class ChromaRadiancePipeline : DiffusionPipelineBase
{
    private readonly T5TextEncoder _t5;
    private readonly ChromaRadianceTransformer _transformer;
    private readonly ChromaRadianceConfig _config;

    /// <summary>Creates a new Chroma Radiance pipeline with all components pre-loaded.</summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="t5">T5-XXL text encoder (joint_attention_dim = 4096, max length 512).</param>
    /// <param name="transformer">Radiance transformer (loaded with <see cref="ChromaRadianceConfig"/>).</param>
    /// <param name="config">Radiance configuration (use <see cref="ChromaRadianceConfig.V1"/> or <c>FromWeights</c>).</param>
    public ChromaRadiancePipeline(IBackend backend, T5TextEncoder t5, ChromaRadianceTransformer transformer,
        ChromaRadianceConfig config)
        : base(backend)
    {
        _t5 = t5;
        _transformer = transformer;
        _config = config;
    }

    /// <summary>Generates an image from pre-tokenized T5 input plus attention masks. API mirrors
    /// <see cref="ChromaPipeline.GenerateFromTokens"/>.</summary>
    /// <param name="promptTokenIdsT5">Prompt token IDs from the T5 tokenizer.</param>
    /// <param name="negativePromptTokenIdsT5">Negative prompt token IDs.</param>
    /// <param name="promptAttentionMaskT5">Tokenizer attention mask for the prompt (1=real token, 0=pad).</param>
    /// <param name="negativeAttentionMaskT5">Tokenizer attention mask for the negative prompt.</param>
    /// <param name="request">Generation parameters. Pass an <see cref="ImageToImageRequest"/> for img2img / inpaint (pixel-space — no VAE encoder needed; strength=0 short-circuits to byte-identical pass-through).</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(
        int[] promptTokenIdsT5,
        int[] negativePromptTokenIdsT5,
        int[] promptAttentionMaskT5,
        int[] negativeAttentionMaskT5,
        TextToImageRequest request,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();

        if (promptAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(promptAttentionMaskT5),
                "Chroma Radiance requires the tokenizer attention mask (1=real, 0=pad) so the pipeline can " +
                "compute the 'first padding token unmasked' rule.");
        if (negativeAttentionMaskT5 is null)
            throw new ArgumentNullException(nameof(negativeAttentionMaskT5),
                "Chroma Radiance requires the tokenizer attention mask for the negative prompt as well.");

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int width = request.Width ?? GenerationDefaults.Chroma.Width;
        int height = request.Height ?? GenerationDefaults.Chroma.Height;
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;
        bool useCfg = cfgScale > 1.0f;

        // Pixel-space: pad dimensions up to the 16-px patch grid, crop the output back at the end.
        int patch = _transformer.PatchSize;
        int padWidth = PadUpTo(width, patch);
        int padHeight = PadUpTo(height, patch);

        // Img2img / inpaint: pixel-space means no VAE — the source image IS the clean sample. Validate
        // against the request resolution (the pad to the patch grid is internal; source + mask are padded
        // alongside the sample below) and handle the strength=0 short-circuit before any model work.
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, height, width, steps);
        bool isImg2Img = request is ImageToImageRequest;
        if (plan.PassThrough)
        {
            Logs.Info("Strength=0; passing source through unchanged");
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), width, height, seed);
        }
        int startStep = plan.StartStep;
        Tensor? maskPixel = plan.MaskPixel;
        bool isMaskedInpaint = maskPixel is not null;

        string opMode = isMaskedInpaint ? $"inpaint (start={startStep}/{steps})"
                      : isImg2Img ? $"img2img (start={startStep}/{steps})"
                      : "txt2img";
        Logs.Info($"Chroma Radiance {opMode}: {width}x{height} (padded {padWidth}x{padHeight}), " +
            $"{steps} steps, cfg={cfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Encode prompts with T5-XXL (identical to ChromaPipeline) ──
        Logs.Info("Encoding text with T5-XXL...");
        Backend.PreloadWeights(_t5.EnumerateWeights());

        int[][] batchT5 = [promptTokenIdsT5];
        int[][] batchMask = [promptAttentionMaskT5];
        Tensor condContext = _t5.Encode(Backend, batchT5, batchMask);

        Tensor? uncondContext = null;
        if (useCfg)
        {
            int[][] negBatchT5 = [negativePromptTokenIdsT5];
            int[][] negBatchMask = [negativeAttentionMaskT5];
            uncondContext = _t5.Encode(Backend, negBatchT5, negBatchMask);
        }

        Tensor condMask = ChromaPipeline.BuildChromaTextMask(promptAttentionMaskT5, batchSize: 1);
        Tensor? uncondMask = useCfg ? ChromaPipeline.BuildChromaTextMask(negativeAttentionMaskT5, batchSize: 1) : null;

        Logs.Info($"Text encoding done in {sw.ElapsedMilliseconds}ms");

        // ── 2. Static-shift flow-match Euler scheduler (shift 1.0 — see config) ──
        TensorShape pixelShape = new TensorShape(1, 3, padHeight, padWidth);
        FlowMatchEulerDiscreteScheduler scheduler = new(_config.SchedulerShift);
        scheduler.SetTimesteps(steps);

        // ── 3. Initial pixel sample (the "latent" IS the image) ──
        // T2I: pure noise scaled by initSigma. Img2img: source padded to the patch grid + AddNoise at
        // sigma[startStep] — no VAE encode, the pixels themselves are the clean sample. Masked inpaint
        // keeps the padded source + padded mask alive for per-step blending (pad fill 1.0 lets the
        // padded border denoise freely; it's cropped away at the end either way).
        Tensor pixels = request.InitialNoise ?? SeedGenerator.CreateNoise(pixelShape, seed);
        if (!pixels.Shape.Equals(pixelShape))
            throw new ArgumentException($"InitialNoise shape {pixels.Shape} != expected {pixelShape}.");
        Tensor? sourcePadded = null;
        Tensor? maskPadded = null;
        if (isImg2Img)
        {
            sourcePadded = PadPixels(((ImageToImageRequest)request).SourceImage, padHeight, padWidth, fill: 0f);
            Tensor noised = new Tensor(pixelShape, DType.F32);
            scheduler.AddNoise(noised, sourcePadded, pixels, startStep);
            pixels.Dispose();
            pixels = noised;
            if (isMaskedInpaint)
            {
                maskPadded = PadPixels(maskPixel!, padHeight, padWidth, fill: 1f);
            }
            else
            {
                sourcePadded.Dispose();
                sourcePadded = null;
            }
        }
        else
        {
            float initSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initSigma - 1.0f) > 1e-6f)
            {
                Tensor scaled = new Tensor(pixelShape, DType.F32);
                Backend.Scale(scaled, pixels, initSigma);
                pixels.Dispose();
                pixels = scaled;
            }
        }

        // ── 4. Denoising loop ──
        // Free T5 (~5 GB) from VRAM before uploading the ~9 GB Radiance transformer; paired re-preload
        // pattern mirrors ChromaPipeline.
        Backend.FreeWeights(_t5.EnumerateWeights());
        Backend.Sync();
        Backend.PreloadWeights(_transformer.EnumerateWeights());

        Logs.Info("Starting Chroma Radiance denoising loop...");
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int condTxtLen = (int)condContext.Shape[1];
        int uncondTxtLen = useCfg ? (int)uncondContext!.Shape[1] : 0;

        for (int i = startStep; i < steps; i++)
        {
            Stopwatch stepSw = Stopwatch.StartNew();
            float sigma = timesteps[i] / 1000.0f;

            // Model predicts x0; convert each pass to velocity BEFORE the CFG combine (ComfyUI order).
            Tensor condX0 = _transformer.Forward(Backend, pixels, condContext, sigma, condTxtLen, condMask);
            Tensor velocity = X0Prediction.ToVelocity(condX0, pixels, sigma);
            condX0.Dispose();

            if (useCfg)
            {
                Tensor uncondX0 = _transformer.Forward(Backend, pixels, uncondContext!, sigma, uncondTxtLen, uncondMask);
                Tensor uncondV = X0Prediction.ToVelocity(uncondX0, pixels, sigma);
                uncondX0.Dispose();

                // VALIDATION-PENDING: Chroma-family cond-anchored CFG (cond + scale*(cond - uncond)) on velocity; verify vs reference.
                Tensor combined = CfgHelper.ApplyCfgCondAnchored(velocity, uncondV, cfgScale);
                uncondV.Dispose();
                velocity.Dispose();
                velocity = combined;
            }

            Tensor newPixels = new Tensor(pixelShape, DType.F32);
            scheduler.Step(newPixels, velocity, pixels, i);
            velocity.Dispose();
            pixels.Dispose();
            pixels = newPixels;

            // Masked-inpaint blend in pixel space: keep unmasked region on the source's
            // flow-matching trajectory by re-noising the source at the next step's sigma.
            // Final step blends with the clean source — no further denoising follows.
            if (maskPadded is not null && sourcePadded is not null)
            {
                int nextStep = i + 1;
                Tensor noisedSource;
                if (nextStep < steps)
                {
                    Tensor freshNoise = SeedGenerator.CreateNoise(pixelShape, seed + nextStep);
                    noisedSource = new Tensor(pixelShape, DType.F32);
                    scheduler.AddNoise(noisedSource, sourcePadded, freshNoise, nextStep);
                    freshNoise.Dispose();
                }
                else
                {
                    noisedSource = sourcePadded;
                }
                MaskBlendUtilities.BlendChannelsInPlace(pixels, noisedSource, maskPadded);
                if (noisedSource != sourcePadded) noisedSource.Dispose();
            }

            stepSw.Stop();
            Logs.Debug($"Step {i + 1}/{steps} (sigma={sigma:F4}) done in {stepSw.ElapsedMilliseconds}ms");
            // Pixel-space preview: the in-flight sample is already an RGB image — no unpack needed.
            onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
            {
                Latent = pixels,
                LatentArch = LatentArchitecture.ChromaRadiance,
            });
        }

        condContext.Dispose();
        condMask.Dispose();
        uncondContext?.Dispose();
        uncondMask?.Dispose();
        sourcePadded?.Dispose();
        maskPadded?.Dispose();

        Backend.Sync();
        Backend.FreeWeights(_transformer.EnumerateWeights());
        Backend.FreeWeights(_t5.EnumerateWeights());

        // ── 5. Crop padding (if any) and convert straight to RGB bytes — no VAE ──
        if (padWidth != width || padHeight != height)
        {
            Tensor cropped = CropPixels(pixels, height, width);
            pixels.Dispose();
            pixels = cropped;
        }

        // Recomposite for masked inpaint: paste the denoised pixels over the source where mask=1.
        // Pixel space has no VAE round-trip drift, but the final blend guarantees the unmasked
        // region is byte-exact to the source (consistent with the latent-space pipelines).
        if (isMaskedInpaint && ((ImageToImageRequest)request).RecompositeAtEnd)
        {
            MaskBlendUtilities.BlendChannelsInPlace(pixels, ((ImageToImageRequest)request).SourceImage, maskPixel!);
        }

        byte[] rgbData = ImagePostProcessor.TensorToRgbBytes(pixels);
        pixels.Dispose();

        sw.Stop();
        Logs.Info($"Chroma Radiance {opMode} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");

        return (rgbData, width, height, seed);
    }

    private static int PadUpTo(int n, int multiple)
    {
        int rem = n % multiple;
        return rem == 0 ? n : n + (multiple - rem);
    }

    /// <summary>Pads a <c>[1, C, H, W]</c> pixel tensor to <c>[1, C, padH, padW]</c> (top-left anchored, matching <see cref="CropPixels"/>) with <paramref name="fill"/> in the new bottom/right border. Source images pad with 0 (mid-gray in [-1, 1]); masks pad with 1 so the cropped-away border denoises freely.</summary>
    private static Tensor PadPixels(Tensor src, int padH, int padW, float fill)
    {
        int channels = (int)src.Shape[1];
        int h = (int)src.Shape[2];
        int w = (int)src.Shape[3];
        if (h == padH && w == padW)
        {
            Tensor clone = new Tensor(src.Shape, DType.F32);
            long bytes = src.Shape.ElementCount * sizeof(float);
            Buffer.MemoryCopy((float*)src.DataPointer, (float*)clone.DataPointer, bytes, bytes);
            return clone;
        }
        Tensor output = new Tensor(new TensorShape(1, channels, padH, padW), DType.F32);
        float* srcPtr = (float*)src.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            long srcPlane = (long)c * h * w;
            long dstPlane = (long)c * padH * padW;
            for (int y = 0; y < padH; y++)
            {
                float* dstRow = dstPtr + dstPlane + (long)y * padW;
                if (y < h)
                {
                    long rowBytes = (long)w * sizeof(float);
                    Buffer.MemoryCopy(srcPtr + srcPlane + (long)y * w, dstRow, rowBytes, rowBytes);
                    for (int x = w; x < padW; x++) dstRow[x] = fill;
                }
                else
                {
                    for (int x = 0; x < padW; x++) dstRow[x] = fill;
                }
            }
        }
        return output;
    }

    /// <summary>Crops a [1, 3, padH, padW] pixel tensor back to the requested [1, 3, H, W] (top-left anchored,
    /// matching ComfyUI's <c>[:, :, :h, :w]</c> un-pad).</summary>
    private static Tensor CropPixels(Tensor padded, int height, int width)
    {
        int padHeight = (int)padded.Shape[2];
        int padWidth = (int)padded.Shape[3];
        Tensor output = new Tensor(new TensorShape(1, 3, height, width), DType.F32);

        float* srcPtr = (float*)padded.DataPointer;
        float* dstPtr = (float*)output.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            long srcPlane = (long)c * padHeight * padWidth;
            long dstPlane = (long)c * height * width;
            for (int y = 0; y < height; y++)
            {
                long rowBytes = (long)width * sizeof(float);
                Buffer.MemoryCopy(
                    srcPtr + srcPlane + (long)y * padWidth,
                    dstPtr + dstPlane + (long)y * width,
                    rowBytes, rowBytes);
            }
        }
        return output;
    }
}
