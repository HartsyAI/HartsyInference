using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan-Animate pipeline driven against the native <see cref="VideoRequest"/>:
/// <see cref="VideoRequest.InitImage"/> is the driving pose/motion input and <c>Extra["AnimateReferenceImage"]</c>
/// carries the character identity image, which is VAE-encoded to the reference latent and (when the checkpoint ships
/// the i2v embedder) CLIP-ViT-H context. Mirrors the SwarmUI backend's <c>WanAnimateLoader.Generate</c>.</summary>
public sealed class WanAnimateRecipePipeline : IVideoRecipePipeline
{
    /// <summary>Request <see cref="VideoRequest.Extra"/> key carrying the character identity image.</summary>
    public const string ReferenceImageKey = "AnimateReferenceImage";

    /// <summary>Wan-Animate motion-encoder input resolution (the face crop is square at this size).</summary>
    private const int MotionEncoderSize = 512;

    /// <summary>ComfyUI's Wan sampling shift — <c>WAN22_Animate</c> inherits <c>WAN21_T2V</c>'s sampling settings.</summary>
    private const float DefaultFlowShift = 8f;

    private readonly IBackend _backend;
    private readonly WanAnimatePipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanAnimateTransformer _transformer;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly ClipVisionEncoder? _clipVision;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Animate pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanAnimateRecipePipeline(IBackend backend, WanAnimatePipeline pipeline, WanVideoConfig config, T5Tokenizer tokenizer,
        T5TextEncoder umt5, WanAnimateTransformer transformer, IWanVaeEncoder vaeEncoder, ClipVisionEncoder? clipVision, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _umt5 = umt5;
        _transformer = transformer;
        _vaeEncoder = vaeEncoder;
        _clipVision = clipVision;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        // TODO(E-IMG-4/5): the driving POSE and FACE clips come from extension-side preprocessors (YOLO11-pose skeleton
        // render + face crop) over a real driving VIDEO; the native contract carries a single frame and no preprocessor
        // seam, so the driving still is tiled to the pose clip and resampled to the motion-encoder square for the face
        // clip — the extension's own "pre-rendered driving input" mode. Background/replace conditioning and
        // continue_motion chunked extension are not modeled.
        ImageData driving = request.InitImage
            ?? throw new InvalidOperationException(
                "Wan-Animate needs a driving pose/motion input in VideoRequest.InitImage.");
        ImageData reference = ResolveReference(request)
            ?? throw new InvalidOperationException(
                $"Wan-Animate needs a character identity image in VideoRequest.Extra[\"{ReferenceImageKey}\"] (an ImageData) — "
                + "the InitImage slot carries the driving pose/motion input.");

        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps > 0 ? request.Steps : _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        if (numFrames < 5)
        {
            throw new InvalidOperationException("Wan-Animate needs at least 5 frames (the face pathway downsamples 4x).");
        }
        float cfgScale = request.CfgScale <= 0 ? _config.GuidanceScale : request.CfgScale;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        Tensor batch = _umt5.Encode(_backend, [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, WanVideoRecipe.TokenLength, _config.TextDim);
        batch.Dispose();
        VideoRecipeUtils.ZeroPaddedRows(promptEmbeds, promptTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(negEmbeds, negTokens, _config.TextDim);
        _backend.Sync();
        _backend.FreeWeights(_umt5.EnumerateWeights());

        Tensor? clipEmbeds = null;
        Tensor? poseClip = null;
        Tensor? faceClip = null;
        Tensor? referenceRgb = null;
        WanAnimateConditioning? conditioning = null;
        try
        {
            if (_clipVision is not null)
            {
                _backend.PreloadWeights(_clipVision.EnumerateWeights());
                ClipImagePreprocessor preprocessor = new ClipImagePreprocessor(imageSize: 224);
                Tensor pixels = preprocessor.Preprocess(reference.Rgb, reference.Width, reference.Height);
                Tensor batched = _clipVision.EncodeHiddenStates(_backend, pixels);
                pixels.Dispose();
                _backend.Sync();
                _backend.FreeWeights(_clipVision.EnumerateWeights());
                Tensor dropped = VideoRecipeUtils.DropBatch(batched);
                batched.Dispose();
                clipEmbeds = VideoRecipeUtils.HostCopy(dropped);
                dropped.Dispose();
            }

            poseClip = VideoRecipeUtils.TileRgbToClip(VideoRecipeUtils.ResizeRgb24(driving, width, height), width, height, numFrames);
            faceClip = VideoRecipeUtils.TileRgbToClip(
                VideoRecipeUtils.ResizeRgb24(driving, MotionEncoderSize, MotionEncoderSize), MotionEncoderSize, MotionEncoderSize, numFrames - 1);
            referenceRgb = VideoRecipeUtils.RgbToReferenceTensor(VideoRecipeUtils.ResizeRgb24(reference, width, height), width, height);

            VideoGenerationRequest inner = new VideoGenerationRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
                FlowShift = DefaultFlowShift,
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            (byte[][] frames, int outW, int outH, int _, WanAnimateConditioning used) = _pipeline.GenerateAnimation(
                promptEmbeds, negEmbeds, referenceRgb, poseClip, faceClip, inner, clipImageEmbeds: clipEmbeds, cachedConditioning: null, onProgress: bridge);
            conditioning = used;
            Logs.Info($"[WanAnimateRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH} ({numFrames}f pose / {numFrames - 1}f face).");
            return VideoRecipeUtils.ToVideoFrames(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanAnimateRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            conditioning?.Dispose();
            poseClip?.Dispose();
            faceClip?.Dispose();
            referenceRgb?.Dispose();
            clipEmbeds?.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <summary>Pulls the character identity image out of the request's arch-specific bag, or null when absent.</summary>
    private static ImageData? ResolveReference(VideoRequest request) =>
        request.Extra.TryGetValue(ReferenceImageKey, out object? value) ? value as ImageData : null;

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _umt5.Dispose();
        _transformer.Dispose();
        (_vaeEncoder as IDisposable)?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
