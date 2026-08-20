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
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan-Animate pipeline driven against the native <see cref="VideoRequest"/>:
/// <see cref="VideoRequest.DrivingVideo"/> (else a tiled <see cref="VideoRequest.InitImage"/>) is the driving
/// pose/motion input, resolved into the pose/face clips by <see cref="WanAnimateDrivingResolver"/>, and
/// <c>Extra["AnimateReferenceImage"]</c> carries the character identity image, which is VAE-encoded to the reference
/// latent and (when the checkpoint ships the i2v embedder) CLIP-ViT-H context. Mirrors the SwarmUI backend's
/// <c>WanAnimateLoader.Generate</c>. <see cref="VideoRequest.AnimateTotalFrames"/> turns the single generation into a
/// chunk loop, each chunk conditioned on the tail of the previous one (ComfyUI <c>continue_motion</c>) and assembled
/// here so trim/boomerang still apply once, to the whole video.</summary>
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
    private readonly ModelAssets.Lora.LoraStack? _loraStack;

    /// <summary>Wraps the constructed Animate pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanAnimateRecipePipeline(IBackend backend, WanAnimatePipeline pipeline, WanVideoConfig config, T5Tokenizer tokenizer,
        T5TextEncoder umt5, WanAnimateTransformer transformer, IWanVaeEncoder vaeEncoder, ClipVisionEncoder? clipVision, List<SafeTensorsLoader> loaders,
        ModelAssets.Lora.LoraStack? loraStack = null)
    {
        _loraStack = loraStack;
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
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (request.DrivingVideo is null && request.InitImage is null)
        {
            throw new InvalidOperationException(
                "Wan-Animate needs a driving motion input: set VideoRequest.DrivingVideo (a driving video whose pose "
                + "skeleton and face crop are auto-derived) or VideoRequest.InitImage (a still tiled across frames).");
        }
        ImageData reference = ResolveReference(request)
            ?? throw new InvalidOperationException(
                $"Wan-Animate needs a character identity image in VideoRequest.Extra[\"{ReferenceImageKey}\"] (an ImageData) — "
                + "the InitImage slot carries the driving pose/motion input.");

        string prompt = request.Prompt;
        // Swarm sends an unset negative as "", not null, so upstream's falsy check is the one to mirror.
        string negative = string.IsNullOrWhiteSpace(request.NegativePrompt)
            ? WanVideoRecipe.DefaultNegativePrompt
            : request.NegativePrompt;
        int steps = request.Steps ?? _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 77, step: _config.VaeTemporalCompression);
        if (numFrames < 5)
        {
            throw new InvalidOperationException("Wan-Animate needs at least 5 frames (the face pathway downsamples 4x).");
        }
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;
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
        Tensor? referenceRgb = null;
        int? drivingFps = null;
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

            referenceRgb = VideoRecipeUtils.RgbToReferenceTensor(VideoRecipeUtils.ResizeRgb24(reference, width, height), width, height);

            // One seed for every chunk: the chunks are separate denoises, so leaving it null would re-roll per chunk
            // and make a run unreproducible. The motion prefix pins continuity regardless of the noise.
            VideoGenerationRequest inner = new VideoGenerationRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = RecipeRequestMapper.MapSeed(request.Seed) ?? SeedGenerator.RandomSeed(),
                FlowShift = DefaultFlowShift,
            };

            List<byte[]> assembled = new List<byte[]>();
            int chunkLen = numFrames;
            int totalFrames = numFrames;
            int motionFrames = WanAnimateChunkMath.SnapMotionFrames(request.AnimateContinueMotionFrames);
            if (motionFrames != request.AnimateContinueMotionFrames)
            {
                Logs.Info($"[WanAnimateRecipePipeline] Motion-context frames {request.AnimateContinueMotionFrames} snapped down to "
                    + $"{motionFrames} (the prefix must be 4n+1, else the trim under-drops and the background clobbers it).");
            }
            int carriedOffset = 0, chunkIndex = 0, plannedChunks = 1;
            int outW = width, outH = height;
            while (true)
            {
                cancel.ThrowIfCancellationRequested();
                // Rewind BEFORE the driving slice, not after: the prefix must be re-driven by the very frames that
                // produced it, which is the whole point of carrying it (ComfyUI decrements video_frame_offset at the
                // top of the node, above every pose/face/background/mask slice).
                int prefix = chunkIndex == 0 ? 0
                    : WanAnimateChunkMath.MotionPrefixFrames(motionFrames, assembled.Count, chunkLen);
                int sliceOffset = WanAnimateChunkMath.SliceOffset(carriedOffset, prefix);
                WanAnimateDrivingResolver.ResolvedClips clips = WanAnimateDrivingResolver.Resolve(
                    _backend, request, width, height, chunkLen, _config.VaeTemporalCompression, MotionEncoderSize, cancel,
                    frameOffset: sliceOffset, pinFrameCount: chunkIndex > 0);
                Tensor? continueClip = null;
                try
                {
                    if (chunkIndex == 0)
                    {
                        chunkLen = clips.FrameCount;
                        drivingFps = clips.DrivingFps;
                        totalFrames = Math.Max(chunkLen, request.AnimateTotalFrames ?? 0);
                        if (totalFrames > chunkLen)
                        {
                            if (motionFrames < 1 || motionFrames >= chunkLen)
                            {
                                throw new InvalidOperationException(
                                    $"Wan-Animate motion-context frames ({motionFrames} after the 4n+1 snap) must be at least 1 and "
                                    + $"shorter than the {chunkLen}-frame chunk, else a continuation chunk adds no new frames.");
                            }
                            plannedChunks = WanAnimateChunkMath.ChunkCount(totalFrames, chunkLen, motionFrames);
                            Logs.Info($"[WanAnimateRecipePipeline] Chunked extension: {totalFrames} frames as {plannedChunks} chunk(s) "
                                + $"of {chunkLen}f with {motionFrames}f motion context.");
                        }
                    }
                    if (prefix > 0)
                    {
                        continueClip = VideoRecipeUtils.PackRgbFramesToClip(
                            assembled.GetRange(assembled.Count - prefix, prefix), width, height);
                    }
                    int stepsBefore = chunkIndex * steps, stepsTotal = plannedChunks * steps;
                    Action<GenerationProgress> bridge = p =>
                    {
                        cancel.ThrowIfCancellationRequested();
                        progress?.Report(new StepPreview { Step = stepsBefore + p.Step, TotalSteps = stepsTotal });
                    };

                    (byte[][] frames, int chunkW, int chunkH, int _, WanAnimateConditioning used, int trimImage) =
                        _pipeline.GenerateAnimation(promptEmbeds, negEmbeds, referenceRgb, clips.PoseClip, clips.FaceClip,
                            inner, clipImageEmbeds: clipEmbeds, cachedConditioning: null,
                            backgroundRgbClip: clips.BackgroundClip, characterMaskClip: clips.MaskClip,
                            continueMotionRgbClip: continueClip, onProgress: bridge);
                    used.Dispose();
                    outW = chunkW;
                    outH = chunkH;
                    for (int i = trimImage; i < frames.Length; i++)
                    {
                        assembled.Add(frames[i]);
                    }
                    Logs.Info($"[WanAnimateRecipePipeline] Chunk {chunkIndex + 1}/{plannedChunks} returned {frames.Length} frames "
                        + $"{chunkW}x{chunkH} (offset {sliceOffset}, {prefix}f context, {trimImage}f trimmed) → {assembled.Count}/{totalFrames}.");
                }
                finally
                {
                    continueClip?.Dispose();
                    clips.PoseClip.Dispose();
                    clips.FaceClip.Dispose();
                    clips.BackgroundClip?.Dispose();
                    clips.MaskClip?.Dispose();
                }
                carriedOffset = WanAnimateChunkMath.NextCarriedOffset(sliceOffset, chunkLen);
                chunkIndex++;
                if (assembled.Count >= totalFrames)
                {
                    break;
                }
            }
            // Frame edits (trim/boomerang) run once over the assembled video, not per chunk.
            byte[][] output = assembled.Count > totalFrames
                ? [.. assembled.GetRange(0, totalFrames)]
                : [.. assembled];
            return VideoRecipeUtils.ToResult(output, outW, outH, request, fps: drivingFps);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanAnimateRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
