using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LongCat-Video pipeline: umT5 prompts, optional first-frame conditioning or video continuation, the flow-match denoising
/// loop, and the Wan 2.1 VAE decode.</summary>
public sealed class LongCatVideoRecipePipeline : IVideoRecipePipeline
{
    /// <summary>The continuation's conditioning: the reference's <c>num_cond_frames</c> default of 13 pixel frames, which is four latents
    /// (<c>run_demo_video_continuation.py</c> line 43; <c>pipeline_longcat_video.py</c> line 280).</summary>
    internal const int ContinuationFrames = 13;
    internal const int ContinuationLatents = 4;

    private readonly IBackend _backend;
    private readonly LongCatVideoPipeline _pipeline;
    private readonly LongCatParts _parts;
    private readonly List<IDisposable> _loaders;
    private readonly bool _distilled;

    /// <summary>Wraps the constructed pipeline, taking ownership of every disposable in <paramref name="loaders"/>. A distilled build runs the
    /// 16-step schedule with its cfg_step LoRA gate on (<paramref name="distilled"/>).</summary>
    internal LongCatVideoRecipePipeline(IBackend backend, LongCatVideoPipeline pipeline, LongCatParts parts, List<IDisposable> loaders,
        bool distilled = false)
    {
        _backend = backend;
        _pipeline = pipeline;
        _parts = parts;
        _loaders = loaders;
        _distilled = distilled;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        // The clip is decoded before any weight is placed, so a bad clip fails before the denoise.
        LongCatContinuation.Source? continuation = request.VideoContinuation is { } clip
            ? PrepareContinuation(clip, cancel)
            : null;
        return GenerateFrom(request, settings, continuation, progress, cancel);
    }

    /// <summary>Generates from a continuation source already prepared (null for none). <see cref="Generate"/> decodes the clip and calls this; the
    /// tests call it directly with a synthetic source.</summary>
    internal VideoGenerationResult GenerateFrom(VideoRequest request, VideoEffectiveSettings settings, LongCatContinuation.Source? continuation,
        IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        // The distilled build's cfg_step LoRA is live for this run only; the undistilled runs see the base weights.
        using IDisposable? cfgStepScope = _distilled ? _parts.CfgStepGate?.Enable(1f) : null;
        cancel.ThrowIfCancellationRequested();
        // A continuation generates at the aspect bucket of its source, not at the requested size (see LongCatContinuation).
        int width = continuation?.GenerationWidth ?? settings.Width;
        int height = continuation?.GenerationHeight ?? settings.Height;
        int frames = settings.Frames;
        int latentFrames = (frames - 1) / LongCatRecipeSupport.TemporalCompressionFactor + 1;
        int latentHeight = height / LongCatRecipeSupport.SpatialCompressionFactor;
        int latentWidth = width / LongCatRecipeSupport.SpatialCompressionFactor;
        if (width % LongCatRecipeSupport.PatchGrid != 0 || height % LongCatRecipeSupport.PatchGrid != 0)
            throw new ArgumentException($"LongCat-Video needs width and height in multiples of {LongCatRecipeSupport.PatchGrid}, got {width}×{height}.");
        int seed = RecipeRequestMapper.MapSeed(settings.Seed) ?? SeedGenerator.RandomSeed();

        (LongCatPrompt positive, LongCatPrompt negative) = LongCatRecipeSupport.EncodePrompts(_backend, _parts, request.Prompt,
            request.NegativePrompt ?? "");
        float[] noise = LongCatRecipeSupport.SegmentNoise(0, seed, _parts.Channels, latentFrames, latentHeight, latentWidth);

        LongCatConditioningMode mode = LongCatConditioningMode.Text;
        int condLatents = 0;
        float[]? condition = null;
        if (request.InitImage is not null)
        {
            // The base reference's I2V init image: diffusers' default resize (a Lanczos stretch to the target size).
            byte[] rgb = FeatureImaging.ResizeRgb24Lanczos(request.InitImage, width, height);
            condition = LongCatRecipeSupport.EncodeFirstFrame(_backend, _parts.Encoder, rgb, width, height, _parts.Channels);
            mode = LongCatConditioningMode.Image;
            condLatents = 1;
        }
        else if (continuation is not null)
        {
            // generate_vc: the source's last 13 frames, resized to the bucket, encoded to four latents that lead the run.
            if (latentFrames <= ContinuationLatents)
                throw new ArgumentException($"A continuation needs more output frames than its {ContinuationFrames}-frame context; got {frames}.");
            condition = LongCatRecipeSupport.EncodeContext(_backend, _parts.Encoder, continuation.ContextFrames, ContinuationLatents, _parts.Channels,
                latentHeight, latentWidth);
            mode = LongCatConditioningMode.Continuation;
            condLatents = ContinuationLatents;
        }

        // The DiT is placed by the streaming scope: resident when it fits, block-streamed when it does not. Each finished step
        // returns the stream's pool reservations (EndStep), which the scope requires.
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);
        float[] latents = LongCatRecipeSupport.RunDenoiseStreamed(_backend, _parts.Dit,
            () => LongCatRecipeSupport.OpenDenoiseStream(_backend, _parts, "longcat-video",
                LongCatRecipeSupport.TokenCount(latentFrames, latentHeight, latentWidth)),
            onStep =>
            {
                LongCatRun run = new(_parts.Channels, latentFrames, latentHeight, latentWidth, noise, positive, negative,
                    settings.Steps, Distill: _distilled, Shift: settings.FlowShift ?? LongCatRecipeSupport.BaseFlowShift, Guidance: settings.CfgScale,
                    Mode: mode, NumCondLatents: condLatents, CondLatents: condition,
                    EnhanceHf: mode == LongCatConditioningMode.Continuation, OnProgress: onStep, Cancel: cancel);
                return _pipeline.Denoise(run).Latents;
            },
            bridge);
        cancel.ThrowIfCancellationRequested();

        byte[][] decoded = LongCatRecipeSupport.DecodeFrames(_backend, _parts.Decoder, latents, _parts.Channels, latentFrames, latentHeight, latentWidth);
        Logs.Info($"[LongCatVideoRecipePipeline] Decoded {decoded.Length} frames {width}x{height}.");
        if (continuation is null)
            return VideoRecipeUtils.ToResult(decoded, width, height, request, audio: null, fps: settings.Fps);

        // The demo's composition (run_demo_video_continuation.py lines 114-116): the source frames as they were, then the generated frames after the
        // context, resized back to the source size (BICUBIC, line 112), written at the target rate.
        List<byte[]> output = [.. continuation.Frames];
        for (int i = ContinuationFrames; i < decoded.Length; i++)
        {
            ImageData generated = new() { Rgb = decoded[i], Width = width, Height = height };
            output.Add(FeatureImaging.ResizeRgb24(generated, continuation.Width, continuation.Height));
        }
        Logs.Info($"[LongCatVideoRecipePipeline] Continuation: {continuation.Frames.Length} source frames, {decoded.Length - ContinuationFrames} generated.");
        return VideoRecipeUtils.ToResult([.. output], continuation.Width, continuation.Height, request, audio: null, fps: LongCatContinuation.TargetFps);
    }

    /// <summary>Decodes the continuation clip at its own size and frame rate, then prepares its source (see <see cref="LongCatContinuation"/>).</summary>
    private static LongCatContinuation.Source PrepareContinuation(VideoClip clip, CancellationToken cancel)
    {
        FfmpegProcessDecoder.Result decoded = new FfmpegProcessDecoder()
            .DecodeAsync(clip.Data, clip.Format, maxFrames: null, scaleWidth: null, scaleHeight: null, cancel: cancel)
            .GetAwaiter().GetResult();
        return LongCatContinuation.Prepare(decoded.Frames.ToArray(), decoded.Width, decoded.Height, decoded.Fps, ContinuationFrames);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
    }
}
