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
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan-Animate-2 pipeline driven against the native <see cref="VideoRequest"/>.
/// <see cref="VideoRequest.DrivingVideo"/> (else a tiled <see cref="VideoRequest.InitImage"/>) supplies the driving
/// motion as <b>raw RGB</b> — there is no pose render, no face crop and no retargeting, so the pose/face/background/
/// mask override clips V1 accepts are refused here by name rather than silently dropped.
/// <c>Extra["AnimateReferenceImage"]</c> carries the character identity image, whose aspect ratio (not the request's)
/// shapes the output; <c>--width</c>/<c>--height</c> are an AREA BUDGET, exactly as upstream's
/// <c>resize_by_area</c> treats them.
///
/// <para>Two umT5 prompts are encoded: <see cref="VideoRequest.Prompt"/> describes the character's appearance and
/// background (upstream tells you NOT to describe motion there), and <c>Extra["AnimateDrivingPrompt"]</c> describes
/// the driving video, defaulting to upstream's boilerplate. The negative prompt applies to the generation stream
/// only — the driving stream is built once, outside the guidance loop.</para></summary>
public sealed class WanAnimate2RecipePipeline : IVideoRecipePipeline
{
    /// <summary>Request <see cref="VideoRequest.Extra"/> key carrying the character identity image, shared with the
    /// V1 recipe so a caller wiring both does not need two keys.</summary>
    public const string ReferenceImageKey = WanAnimateRecipePipeline.ReferenceImageKey;

    /// <summary>Request <see cref="VideoRequest.Extra"/> key carrying the driving stream's own umT5 prompt.</summary>
    public const string DrivingPromptKey = "AnimateDrivingPrompt";

    /// <summary>Upstream's default <c>prompt_ref</c> — "reference video of the character's motion".</summary>
    public const string DefaultDrivingPrompt = "人物动作的参考视频";

    /// <summary>Output dimensions are multiples of this (upstream <c>resize_by_area(divisor=16)</c>), which is also
    /// the VAE stride times the patch size, so the token grid is exact.</summary>
    private const int SizeDivisor = 16;

    /// <summary>The only solver this pipeline runs; the reference's FlowDPM++ 2M is not ported.</summary>
    private const string SupportedSampler = WanAnimate2Pipeline.SubstitutedSampler;

    /// <summary>Source frames decoded per output frame requested, before fps resampling. Bounds the decode of a long
    /// driving video: a source above this multiple of the target rate is truncated rather than materialized.</summary>
    private const int MaxSourceFpsRatio = 6;

    private readonly IBackend _backend;
    private readonly WanAnimate2Pipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanAnimate2Transformer _transformer;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly ClipVisionEncoder _clipVision;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly ModelAssets.Lora.LoraStack? _loraStack;

    /// <summary>Wraps the constructed Animate-2 pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanAnimate2RecipePipeline(IBackend backend, WanAnimate2Pipeline pipeline, WanVideoConfig config,
        T5Tokenizer tokenizer, T5TextEncoder umt5, WanAnimate2Transformer transformer, IWanVaeEncoder vaeEncoder,
        ClipVisionEncoder clipVision, List<SafeTensorsLoader> loaders, ModelAssets.Lora.LoraStack? loraStack = null)
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
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        Validate(request);
        ImageData reference = ResolveReference(request)
            ?? throw new InvalidOperationException(
                $"Wan-Animate-2 needs a character identity image in VideoRequest.Extra[\"{ReferenceImageKey}\"] (an ImageData) — "
                + "the InitImage slot carries the driving motion input.");

        // The CLI width/height are an area budget; the OUTPUT aspect follows the reference image, and the driving
        // video is force-resized into that same geometry (ComfyUI's simplification of upstream's per-input
        // resize_by_area, which the spec endorses as the case to port first).
        (int requestedWidth, int requestedHeight) = VideoRecipeUtils.ResolveResolution(request, SizeDivisor);
        (int width, int height) = ShapeToAreaBudget((long)requestedWidth * requestedHeight, reference.Width, reference.Height);
        if (width != requestedWidth || height != requestedHeight)
        {
            Logs.Info($"[WanAnimate2RecipePipeline] {requestedWidth}x{requestedHeight} is an area budget "
                + $"({(long)requestedWidth * requestedHeight} px); the reference image's {reference.Width}x{reference.Height} "
                + $"aspect shapes the output to {width}x{height}.");
        }

        int steps = request.Steps ?? _config.NumInferenceSteps;
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;
        int targetFps = request.Fps is > 0 ? request.Fps.Value : 24;
        int chunkLen = VideoRecipeUtils.ResolveFrames(request, modelDefault: WanAnimate2Pipeline.ClipLength,
            step: _config.VaeTemporalCompression);
        if (chunkLen < 5)
        {
            throw new InvalidOperationException($"Wan-Animate-2 needs at least 5 frames per chunk; got {chunkLen}.");
        }

        string negative = string.IsNullOrWhiteSpace(request.NegativePrompt)
            ? WanVideoRecipe.DefaultNegativePrompt
            : request.NegativePrompt;
        string drivingPrompt = ResolveDrivingPrompt(request);

        // Three contexts, one batched encode: the generation stream's prompt, its negative, and the driving
        // stream's own prompt. Only these are hoisted out of the chunk loop.
        int[] promptTokens = _tokenizer.Encode(request.Prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        int[] drivingTokens = _tokenizer.Encode(drivingPrompt);
        Tensor batch = _umt5.Encode(_backend, [promptTokens, negTokens, drivingTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens),
             T5Tokenizer.CreateAttentionMask(drivingTokens)]);
        Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, WanVideoRecipe.TokenLength, _config.TextDim);
        Tensor drivingEmbeds = CfgHelper.SliceBatchElement(batch, 2, WanVideoRecipe.TokenLength, _config.TextDim);
        batch.Dispose();
        VideoRecipeUtils.ZeroPaddedRows(promptEmbeds, promptTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(negEmbeds, negTokens, _config.TextDim);
        VideoRecipeUtils.ZeroPaddedRows(drivingEmbeds, drivingTokens, _config.TextDim);
        _backend.Sync();
        _backend.FreeWeights(_umt5.EnumerateWeights());

        Tensor? referenceRgb = null, referenceClip = null;
        try
        {
            List<byte[]> drivingFrames = ResolveDrivingFrames(request, width, height, chunkLen, targetFps, cancel);
            int available = drivingFrames.Count;
            int totalFrames = Math.Min(request.AnimateTotalFrames ?? chunkLen, available);
            totalFrames = Math.Max(totalFrames, 5);

            referenceRgb = VideoRecipeUtils.RgbToReferenceTensor(
                VideoRecipeUtils.LetterboxRgb24(reference, width, height), width, height);
            referenceClip = EncodeClipVision(reference.Rgb, reference.Width, reference.Height);

            // One seed for the whole request: the chunks are separate denoises, so leaving it null would re-roll per
            // chunk and make a run unreproducible. Upstream draws fresh noise per chunk; offsetting by the chunk
            // index keeps that independence while staying deterministic.
            int baseSeed = RecipeRequestMapper.MapSeed(request.Seed) ?? SeedGenerator.RandomSeed();
            int plannedChunks = PlanChunkCount(totalFrames, chunkLen);
            if (plannedChunks > 1)
            {
                Logs.Info($"[WanAnimate2RecipePipeline] Chunked: {totalFrames} frames as ~{plannedChunks} chunk(s) of "
                    + $"{chunkLen}f with a {WanAnimate2Pipeline.ChunkOverlapFrames}-frame carry.");
            }

            List<byte[]> assembled = new List<byte[]>();
            byte[]? carriedFrame = null;
            int start = 0, chunkIndex = 0, outW = width, outH = height;
            while (assembled.Count < totalFrames)
            {
                cancel.ThrowIfCancellationRequested();
                int chunkFrames = SnapChunkLength(Math.Min(chunkLen, available - start), _config.VaeTemporalCompression);
                if (chunkFrames < 5)
                {
                    break;
                }
                List<byte[]> slice = drivingFrames.GetRange(start, Math.Min(chunkFrames, available - start));
                WanAnimateDrivingResolver.FitFrames(slice, chunkFrames);

                Tensor drivingClip = VideoRecipeUtils.PackRgbFramesToClip(slice, width, height);
                Tensor? drivingClipEmbeds = null;
                Tensor? carriedTensor = null;
                try
                {
                    // Recomputed per chunk: the driving stream's image context is THIS chunk's frame 0, not the
                    // reference image's, and the frame changes as start advances.
                    drivingClipEmbeds = EncodeClipVision(slice[0], width, height);
                    if (carriedFrame is not null)
                    {
                        carriedTensor = VideoRecipeUtils.RgbToReferenceTensor(carriedFrame, width, height);
                    }
                    int stepsBefore = chunkIndex * steps, stepsTotal = Math.Max(plannedChunks, chunkIndex + 1) * steps;
                    Action<GenerationProgress> bridge = p =>
                    {
                        cancel.ThrowIfCancellationRequested();
                        progress?.Report(new StepPreview { Step = stepsBefore + p.Step, TotalSteps = stepsTotal });
                    };
                    VideoGenerationRequest inner = new VideoGenerationRequest
                    {
                        Prompt = request.Prompt,
                        NegativePrompt = negative,
                        Width = width,
                        Height = height,
                        Steps = steps,
                        CfgScale = cfgScale,
                        Seed = baseSeed + chunkIndex,
                        FlowShift = request.FlowShift ?? WanAnimate2Pipeline.DefaultFlowShift,
                    };
                    (byte[][] frames, int chunkW, int chunkH, int _) = _pipeline.GenerateChunk(
                        promptEmbeds, negEmbeds, drivingEmbeds, referenceRgb, drivingClip, inner,
                        referenceClipEmbeds: referenceClip, drivingClipEmbeds: drivingClipEmbeds,
                        carriedRgbFrame: carriedTensor, onProgress: bridge);
                    outW = chunkW;
                    outH = chunkH;
                    // Chunk > 0 re-renders the carried frame; upstream drops exactly first_num leading frames.
                    int trim = chunkIndex == 0 ? 0 : WanAnimate2Pipeline.ChunkOverlapFrames;
                    for (int i = trim; i < frames.Length; i++)
                    {
                        assembled.Add(frames[i]);
                    }
                    carriedFrame = assembled.Count > 0 ? assembled[^1] : null;
                    Logs.Info($"[WanAnimate2RecipePipeline] Chunk {chunkIndex + 1} returned {frames.Length} frames "
                        + $"{chunkW}x{chunkH} (driving offset {start}, {trim}f trimmed) → {assembled.Count}/{totalFrames}.");
                }
                finally
                {
                    carriedTensor?.Dispose();
                    drivingClipEmbeds?.Dispose();
                    drivingClip.Dispose();
                }
                int stride = chunkFrames - WanAnimate2Pipeline.ChunkOverlapFrames;
                if (stride <= 0 || start + chunkFrames >= available)
                {
                    break;
                }
                start += stride;
                chunkIndex++;
            }
            if (assembled.Count == 0)
            {
                throw new InvalidOperationException("Wan-Animate-2 produced no frames — the driving input decoded to nothing usable.");
            }
            // Frame edits (trim/boomerang) run once over the assembled video, not per chunk.
            byte[][] output = assembled.Count > totalFrames ? [.. assembled.GetRange(0, totalFrames)] : [.. assembled];
            return VideoRecipeUtils.ToResult(output, outW, outH, request, fps: targetFps);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanAnimate2RecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            referenceClip?.Dispose();
            referenceRgb?.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
            drivingEmbeds.Dispose();
        }
    }

    /// <summary>Rejects, by name, every conditioning object Animate-2 physically cannot consume. V1 accepted these
    /// because it had a pose/face/background/mask pathway; Animate-2 deleted all of it, so accepting them would
    /// produce a plausible clip with the user's input discarded.</summary>
    private static void Validate(VideoRequest request)
    {
        if (request.DrivingVideo is null && request.InitImage is null)
        {
            throw new InvalidOperationException(
                "Wan-Animate-2 needs a driving motion input: set VideoRequest.DrivingVideo (raw video — no pose or "
                + "face preprocessing is applied) or VideoRequest.InitImage (a still tiled across frames).");
        }
        List<string> unsupported = [];
        if (request.DrivingPoseVideo is not null) unsupported.Add(nameof(VideoRequest.DrivingPoseVideo));
        if (request.DrivingFaceVideo is not null) unsupported.Add(nameof(VideoRequest.DrivingFaceVideo));
        if (request.DrivingBackgroundVideo is not null) unsupported.Add(nameof(VideoRequest.DrivingBackgroundVideo));
        if (request.DrivingMaskVideo is not null) unsupported.Add(nameof(VideoRequest.DrivingMaskVideo));
        if (unsupported.Count > 0)
        {
            throw new NotSupportedException(
                $"Wan-Animate-2 has no pose, face, background or character-mask pathway — {string.Join(", ", unsupported)} "
                + "cannot be applied. Those are Wan-Animate (V1) inputs; the driving video is consumed as raw RGB here.");
        }
        if (!string.IsNullOrWhiteSpace(request.Sampler)
            && !string.Equals(request.Sampler, SupportedSampler, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Wan-Animate-2 has no '{request.Sampler}' implementation. Upstream samples with FlowDPM++ 2M midpoint, "
                + "which this engine does not implement; UniPC stands in. Clear the sampler or set it to 'unipc'.");
        }
    }

    /// <summary>The driving stream's umT5 prompt, or upstream's boilerplate when the caller supplied none.</summary>
    private static string ResolveDrivingPrompt(VideoRequest request)
    {
        if (request.Extra.TryGetValue(DrivingPromptKey, out object? value) && value is string text && !string.IsNullOrWhiteSpace(text))
        {
            return text;
        }
        return DefaultDrivingPrompt;
    }

    /// <summary>Decodes the driving video (or tiles the init-image still) into raw RGB frames at
    /// <paramref name="targetFps"/>, letterboxed into the output geometry. No skeleton, no face crop — the frames go
    /// straight to the VAE.</summary>
    private static List<byte[]> ResolveDrivingFrames(VideoRequest request, int width, int height, int chunkLen,
        int targetFps, CancellationToken cancel)
    {
        if (request.DrivingVideo is null)
        {
            ImageData still = request.InitImage!;
            Logs.Info($"[WanAnimate2RecipePipeline] No driving video — tiling the init image across {chunkLen} frames.");
            byte[] resized = VideoRecipeUtils.LetterboxRgb24(still, width, height);
            List<byte[]> tiled = new List<byte[]>(chunkLen);
            for (int i = 0; i < chunkLen; i++)
            {
                tiled.Add(resized);
            }
            return tiled;
        }
        int wanted = Math.Max(chunkLen, request.AnimateTotalFrames ?? chunkLen);
        int maxSource = wanted * MaxSourceFpsRatio;
        FfmpegProcessDecoder.Result decoded;
        try
        {
            decoded = new FfmpegProcessDecoder()
                .DecodeAsync(request.DrivingVideo.Data, request.DrivingVideo.Format, maxSource, width, height, cancel, letterbox: true)
                .GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Logs.Error($"[WanAnimate2RecipePipeline] Failed to decode the driving video (format hint '{request.DrivingVideo.Format ?? "none"}').", ex);
            throw;
        }
        if (decoded.Frames.Count == 0)
        {
            throw new InvalidOperationException("The Wan-Animate-2 driving video decoded to zero frames.");
        }
        List<byte[]> resampled = ResampleToFps(decoded.Frames, decoded.Fps, targetFps);
        Logs.Info($"[WanAnimate2RecipePipeline] Driving video decoded: {decoded.Frames.Count} frame(s) @ {decoded.Fps:0.##} fps "
            + $"→ {resampled.Count} frame(s) @ {targetFps} fps at {width}x{height}.");
        return resampled;
    }

    /// <summary>Upstream's <c>get_frame_indices</c>: a nearest-frame resample, <c>idx = round(i / targetFps · srcFps)</c>
    /// clamped into range, over <c>floor(count / srcFps · targetFps)</c> output frames. A non-positive or unknown
    /// source rate passes the frames through 1:1.</summary>
    internal static List<byte[]> ResampleToFps(List<byte[]> frames, double sourceFps, int targetFps)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (targetFps <= 0 || sourceFps <= 0.5 || Math.Abs(sourceFps - targetFps) < 0.01)
        {
            return frames;
        }
        int targetCount = (int)(frames.Count / sourceFps * targetFps);
        if (targetCount < 1)
        {
            targetCount = 1;
        }
        List<byte[]> output = new List<byte[]>(targetCount);
        for (int i = 0; i < targetCount; i++)
        {
            int idx = (int)Math.Round(i / (double)targetFps * sourceFps, MidpointRounding.AwayFromZero);
            output.Add(frames[Math.Clamp(idx, 0, frames.Count - 1)]);
        }
        return output;
    }

    /// <summary>Snaps a chunk down onto the causal VAE's <c>step·n + 1</c> grid. Upstream's <c>zigzag_padding</c>
    /// instead pads the tail up by ping-ponging through the frames; both only affect frames that get trimmed, and
    /// ComfyUI takes the same snap-and-hold route.</summary>
    internal static int SnapChunkLength(int frames, int temporalStep) =>
        frames < 1 ? 0 : 1 + (frames - 1) / temporalStep * temporalStep;

    /// <summary>Chunks needed for <paramref name="totalFrames"/> at a stride of <c>chunkLen - 1</c>.</summary>
    internal static int PlanChunkCount(int totalFrames, int chunkLen)
    {
        int stride = chunkLen - WanAnimate2Pipeline.ChunkOverlapFrames;
        if (stride <= 0 || totalFrames <= chunkLen)
        {
            return 1;
        }
        return 1 + (int)Math.Ceiling((totalFrames - chunkLen) / (double)stride);
    }

    /// <summary>Upstream's <c>calculate_new_size</c>: the largest <c>(w, h)</c> whose product fits
    /// <paramref name="areaBudget"/>, both multiples of 16, minimising the deviation from the SOURCE aspect ratio.
    /// This is why <c>--width 720 --height 1280</c> means "budget 921600 px" and not "produce 720x1280".</summary>
    internal static (int Width, int Height) ShapeToAreaBudget(long areaBudget, int sourceWidth, int sourceHeight)
    {
        if (areaBudget < SizeDivisor * SizeDivisor || sourceWidth < 1 || sourceHeight < 1)
        {
            return (SizeDivisor, SizeDivisor);
        }
        double aspect = sourceWidth / (double)sourceHeight;
        int bestW = SizeDivisor, bestH = SizeDivisor;
        double bestScore = double.MaxValue;
        long bestArea = 0;
        int centre = (int)Math.Round(Math.Sqrt(areaBudget / aspect) / SizeDivisor);
        for (int step = Math.Max(1, centre - 4); step <= centre + 4; step++)
        {
            int h = step * SizeDivisor;
            int w = Math.Max(SizeDivisor, (int)Math.Round(h * aspect / SizeDivisor) * SizeDivisor);
            if ((long)w * h > areaBudget)
            {
                continue;
            }
            double score = Math.Abs(w / (double)h - aspect);
            // Ties go to the larger frame: two candidates can hit the same aspect deviation and the budget exists
            // to be spent.
            if (score < bestScore - 1e-9 || (Math.Abs(score - bestScore) <= 1e-9 && (long)w * h > bestArea))
            {
                bestScore = score;
                bestArea = (long)w * h;
                bestW = w;
                bestH = h;
            }
        }
        return (bestW, bestH);
    }

    /// <summary>CLIP-ViT-H penultimate hidden state of one RGB24 frame, host-materialized — 257 image tokens that get
    /// prepended to that stream's umT5 context.</summary>
    private Tensor EncodeClipVision(byte[] rgb24, int width, int height)
    {
        _backend.PreloadWeights(_clipVision.EnumerateWeights());
        ClipImagePreprocessor preprocessor = new ClipImagePreprocessor(imageSize: 224);
        Tensor pixels = preprocessor.Preprocess(rgb24, width, height);
        Tensor batched = _clipVision.EncodeHiddenStates(_backend, pixels);
        pixels.Dispose();
        _backend.Sync();
        _backend.FreeWeights(_clipVision.EnumerateWeights());
        Tensor dropped = VideoRecipeUtils.DropBatch(batched);
        batched.Dispose();
        Tensor host = VideoRecipeUtils.HostCopy(dropped);
        dropped.Dispose();
        return host;
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
