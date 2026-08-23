using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
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

/// <summary>A constructed Wan-Animate pipeline driven against the native <see cref="VideoRequest"/>: <see cref="VideoRequest.DrivingVideo"/> (else a tiled <see cref="VideoRequest.InitImage"/>) is the driving pose/motion input, resolved into the pose/face clips by <see cref="WanAnimateDrivingResolver"/>, and <c>Extra["AnimateReferenceImage"]</c> carries the character identity image, which is VAE-encoded to the reference latent and (when the checkpoint ships the i2v embedder) CLIP-ViT-H context. Mirrors the SwarmUI backend's <c>WanAnimateLoader.Generate</c>. <see cref="VideoRequest.AnimateTotalFrames"/> turns the single generation into a chunk loop, each chunk conditioned on the tail of the previous one (ComfyUI <c>continue_motion</c>) and assembled here so trim/boomerang still apply once, to the whole video.</summary>
public sealed class WanAnimateRecipePipeline : IVideoRecipePipeline
{
    /// <summary>Request <see cref="VideoRequest.Extra"/> key carrying the character identity image.</summary>
    public const string ReferenceImageKey = "AnimateReferenceImage";

    /// <summary>Wan-Animate motion-encoder input resolution (the face crop is square at this size).</summary>
    private const int MotionEncoderSize = 512;

    /// <summary>Upstream's <c>animate_14B.sample_shift</c>.</summary>
    private const float DefaultFlowShift = 5f;

    /// <summary>The only solver <see cref="WanAnimatePipeline"/> runs; upstream's other choice (<c>dpm++</c>) is not ported.</summary>
    private const string SupportedSampler = "unipc";

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
    private WanAnimateConditioning? _conditioningCache;
    private string? _conditioningCacheKey;

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
        if (!string.IsNullOrWhiteSpace(request.Sampler)
            && !string.Equals(request.Sampler, SupportedSampler, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Wan-Animate samples with UniPC and has no '{request.Sampler}' implementation — upstream's other "
                + "choice, dpm++ (FlowDPMSolverMultistep), isn't ported. Clear the sampler or set it to 'unipc'.");
        }
        // Refused by name at any non-default value, not silently ignored: these scale attention pathways only the
        // Animate-2 frame-local attention has.
        List<string> unsupportedStrengths = [];
        if (request.AnimatePoseStrength is double ps && ps != 1.0) unsupportedStrengths.Add($"{nameof(VideoRequest.AnimatePoseStrength)}={ps}");
        if (request.AnimateReferenceImageStrength is double rs && rs != 1.0) unsupportedStrengths.Add($"{nameof(VideoRequest.AnimateReferenceImageStrength)}={rs}");
        if (unsupportedStrengths.Count > 0)
        {
            throw new NotSupportedException(
                $"Wan-Animate (V1) has no pose or reference-image attention-strength pathway — {string.Join(", ", unsupportedStrengths)} "
                + "are Wan-Animate-2 controls and cannot be applied here. Leave them at 1.0 (or untoggled), or select an Animate-2 checkpoint.");
        }
        // A sigma schedule needs the Euler seam to re-space; UniPC owns its own spacing, so a named schedule cannot
        // be honored here either and is refused rather than dropped.
        if (!string.IsNullOrWhiteSpace(request.Scheduler))
        {
            throw new NotSupportedException(
                $"Sigma schedule '{request.Scheduler}' is not available on Wan-Animate — the family samples with "
                + "UniPC, which owns its own sigma spacing. Leave the schedule unset.");
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
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request,
            VideoRecipeUtils.PatchAlignedMultiple(_config.VaeSpatialCompression, _config.PatchSize));

        string cacheKey = BuildConditioningKey(request, reference, width, height, numFrames);

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        (Tensor promptEmbeds, Tensor negEmbeds) = VideoRecipeUtils.EncodeWanPrompts(
            _backend, _umt5, _config.TextDim, promptTokens, negTokens);

        Tensor? clipEmbeds = null;
        Tensor? referenceRgb = null;
        int? drivingFps = null;
        try
        {
            if (_clipVision is not null)
            {
                clipEmbeds = VideoRecipeUtils.EncodeClipVision(_backend, _clipVision, reference.Rgb, reference.Width, reference.Height);
            }

            referenceRgb = VideoRecipeUtils.RgbToReferenceTensor(VideoRecipeUtils.LetterboxRgb24(reference, width, height), width, height);

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
                FlowShift = request.FlowShift ?? DefaultFlowShift,
            };

            // Anti-drift stats come from the reference image's OWN pixels — content only, never the letterboxed
            // canvas: black pad bars drag the mean dark, which is this correction's known failure mode.
            float colorStrength = (float)(request.AnimateColorCorrection ?? 1.0);
            VideoColorMatch.LabStats refColorStats = colorStrength > 0f
                ? VideoColorMatch.ComputeStats(reference.Rgb, reference.Width, reference.Height)
                : default;

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

                    // Only chunk 0 is cacheable: a continuation chunk's conditioning carries that chunk's motion
                    // prefix and its own driving slice.
                    bool cacheable = chunkIndex == 0;
                    WanAnimateConditioning? cached = cacheable && string.Equals(cacheKey, _conditioningCacheKey, StringComparison.Ordinal)
                        ? _conditioningCache : null;
                    (byte[][] frames, int chunkW, int chunkH, int _, WanAnimateConditioning used, int trimImage) =
                        _pipeline.GenerateAnimation(promptEmbeds, negEmbeds, referenceRgb, clips.PoseClip, clips.FaceClip,
                            inner, clipImageEmbeds: clipEmbeds, cachedConditioning: cached,
                            backgroundRgbClip: clips.BackgroundClip, characterMaskClip: clips.MaskClip,
                            continueMotionRgbClip: continueClip, onProgress: bridge);
                    StoreConditioning(cacheable, cacheKey, used);
                    outW = chunkW;
                    outH = chunkH;
                    // Before the trim/append, so the emitted frames and the carried motion prefix are fixed in one pass.
                    VideoRecipeUtils.CorrectContinuationChunk(frames, chunkW, chunkH, chunkIndex, refColorStats, colorStrength);
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

    /// <summary>Keeps a chunk-0 conditioning for the next generation with identical inputs, disposing whatever it replaces; anything else (a continuation chunk) is disposed outright.</summary>
    private void StoreConditioning(bool cacheable, string key, WanAnimateConditioning conditioning)
    {
        if (!cacheable)
        {
            conditioning.Dispose();
            return;
        }
        if (ReferenceEquals(_conditioningCache, conditioning))
        {
            return;
        }
        _conditioningCache?.Dispose();
        _conditioningCache = conditioning;
        _conditioningCacheKey = key;
    }

    /// <summary>Content key for the cross-generation conditioning cache — every input the VAE + motion encode reads. A hit derives its geometry from the cached pose latent, so anything that can move the geometry or the encoded pixels must be in here or a stale entry would silently generate at the wrong size.</summary>
    private static string BuildConditioningKey(VideoRequest request, ImageData reference, int width, int height, int frames)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> header = stackalloc byte[24];
        BinaryPrimitives.WriteInt32LittleEndian(header, width);
        BinaryPrimitives.WriteInt32LittleEndian(header[4..], height);
        BinaryPrimitives.WriteInt32LittleEndian(header[8..], frames);
        BinaryPrimitives.WriteInt32LittleEndian(header[12..], reference.Width);
        BinaryPrimitives.WriteInt32LittleEndian(header[16..], reference.Height);
        BinaryPrimitives.WriteInt32LittleEndian(header[20..], request.DrivingAutoPreprocess ? 1 : 0);
        hash.AppendData(header);
        hash.AppendData(reference.Rgb);
        AppendImage(hash, request.InitImage);   // the driving fallback when there is no driving video
        AppendClip(hash, request.DrivingVideo);
        AppendClip(hash, request.DrivingPoseVideo);
        AppendClip(hash, request.DrivingFaceVideo);
        AppendClip(hash, request.DrivingBackgroundVideo);
        AppendClip(hash, request.DrivingMaskVideo);
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary>Marks an optional input present or absent, so an absent slot can never alias onto the next slot's bytes.</summary>
    private static void AppendPresence(IncrementalHash hash, bool present)
    {
        Span<byte> marker = stackalloc byte[1];
        marker[0] = present ? (byte)1 : (byte)0;
        hash.AppendData(marker);
    }

    private static void AppendImage(IncrementalHash hash, ImageData? image)
    {
        AppendPresence(hash, image is not null);
        if (image is null)
        {
            return;
        }
        Span<byte> size = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(size, image.Width);
        BinaryPrimitives.WriteInt32LittleEndian(size[4..], image.Height);
        hash.AppendData(size);
        hash.AppendData(image.Rgb);
    }

    private static void AppendClip(IncrementalHash hash, VideoClip? clip)
    {
        AppendPresence(hash, clip is not null);
        if (clip is null)
        {
            return;
        }
        hash.AppendData(Encoding.UTF8.GetBytes(clip.Format ?? ""));
        AppendPresence(hash, false);
        hash.AppendData(clip.Data);
    }

    /// <summary>Pulls the character identity image out of the request's arch-specific bag, or null when absent.</summary>
    private static ImageData? ResolveReference(VideoRequest request) =>
        request.Extra.TryGetValue(ReferenceImageKey, out object? value) ? value as ImageData : null;

    /// <inheritdoc/>
    public void Dispose()
    {
        _conditioningCache?.Dispose();
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
