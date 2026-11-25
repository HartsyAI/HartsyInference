using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LongCat-Video pipeline: umT5 prompts, optional first-frame conditioning, the flow-match denoising loop, and the
/// Wan 2.1 VAE decode.</summary>
public sealed class LongCatVideoRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly LongCatVideoPipeline _pipeline;
    private readonly LongCatParts _parts;
    private readonly List<IDisposable> _loaders;

    /// <summary>Wraps the constructed pipeline, taking ownership of every disposable in <paramref name="loaders"/>.</summary>
    internal LongCatVideoRecipePipeline(IBackend backend, LongCatVideoPipeline pipeline, LongCatParts parts, List<IDisposable> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _parts = parts;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int width = settings.Width, height = settings.Height, frames = settings.Frames;
        int latentFrames = (frames - 1) / LongCatRecipeSupport.TemporalCompressionFactor + 1;
        int latentHeight = height / LongCatRecipeSupport.SpatialCompressionFactor;
        int latentWidth = width / LongCatRecipeSupport.SpatialCompressionFactor;
        if (width % LongCatRecipeSupport.PatchGrid != 0 || height % LongCatRecipeSupport.PatchGrid != 0)
            throw new ArgumentException($"LongCat-Video needs width and height in multiples of {LongCatRecipeSupport.PatchGrid}, got {width}×{height}.");
        int seed = RecipeRequestMapper.MapSeed(settings.Seed) ?? SeedGenerator.RandomSeed();

        (LongCatPrompt positive, LongCatPrompt negative) = LongCatRecipeSupport.EncodePrompts(_backend, _parts, request.Prompt,
            request.NegativePrompt ?? "");
        float[] noise = LongCatRecipeSupport.Noise(seed, _parts.Channels, latentFrames, latentHeight, latentWidth);

        LongCatConditioningMode mode = LongCatConditioningMode.Text;
        int condLatents = 0;
        float[]? firstFrame = null;
        if (request.InitImage is not null)
        {
            byte[] rgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
            firstFrame = LongCatRecipeSupport.EncodeFirstFrame(_backend, _parts.Encoder, rgb, width, height, _parts.Channels);
            mode = LongCatConditioningMode.Image;
            condLatents = 1;
        }

        // The DiT is placed by the streaming scope: resident when it fits, block-streamed when it does not. Each finished step
        // returns the stream's pool reservations (EndStep), which the scope requires.
        float[] latents;
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);
        using (BlockStreamingScope stream = LongCatRecipeSupport.OpenDenoiseStream(_backend, _parts, "longcat-video"))
        {
            LongCatRun run = new(_parts.Channels, latentFrames, latentHeight, latentWidth, noise, positive, negative,
                settings.Steps, Distill: false, Shift: settings.FlowShift ?? LongCatRecipeSupport.BaseFlowShift, Guidance: settings.CfgScale,
                Mode: mode, NumCondLatents: condLatents, CondLatents: firstFrame,
                OnProgress: step => { stream.EndStep(); bridge(step); }, Cancel: cancel);
            latents = _pipeline.Denoise(run).Latents;
        }
        // The VAE decode runs next, so the DiT leaves the device now (the scope has already evicted any streamed blocks).
        _backend.Sync();
        _backend.FreeWeights(_parts.Dit.EnumerateWeights());
        cancel.ThrowIfCancellationRequested();

        byte[][] decoded = LongCatRecipeSupport.DecodeFrames(_backend, _parts.Decoder, latents, _parts.Channels, latentFrames, latentHeight, latentWidth);
        Logs.Info($"[LongCatVideoRecipePipeline] Decoded {decoded.Length} frames {width}x{height}.");
        return VideoRecipeUtils.ToResult(decoded, width, height, request, audio: null, fps: settings.Fps);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable loader in _loaders)
            loader.Dispose();
        _loaders.Clear();
    }
}
