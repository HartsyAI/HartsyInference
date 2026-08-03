using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed MiniMax-H3 pipeline driven from the native <see cref="VideoRequest"/>. H3 emits a stereo
/// soundtrack with every clip, so the result carries both streams.</summary>
public sealed class MiniMaxH3RecipePipeline : IVideoRecipePipeline
{
    private readonly MiniMaxH3Pipeline _pipeline;
    private readonly MiniMaxH3Config _config;
    private readonly IBackend _backend;
    private readonly MiniMaxH3TextEncoder _textEncoder;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Takes ownership of the pipeline, the pre-encoded conditioning, and every loader backing the weights.</summary>
    public MiniMaxH3RecipePipeline(IBackend backend, MiniMaxH3Pipeline pipeline, MiniMaxH3Config config,
        MiniMaxH3TextEncoder textEncoder, Qwen2Tokenizer tokenizer, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int fps = request.Fps ?? MiniMaxH3Geometry.Fps;
        int requestedFrames = request.Frames ?? 124;
        // H3's grids are coarse and non-obvious: frames snap to 17k+5, latent frames are NOT frames/4, and each pixel
        // axis rounds to 32 (a multiple of 16 alone leaves an odd latent axis and the 2x2 patchifier drops its last
        // row/column). Audio length follows the ALIGNED frame count so the two streams end together.
        int frames = MiniMaxH3Geometry.AlignFrameCount(requestedFrames);
        int width = MiniMaxH3Geometry.Round(request.Width ?? 1344);
        int height = MiniMaxH3Geometry.Round(request.Height ?? 768);
        if (frames != requestedFrames || width != (request.Width ?? 1344) || height != (request.Height ?? 768))
        {
            Logs.Info($"[MiniMaxH3RecipePipeline] Geometry snapped to H3's grid: "
                + $"{request.Width}x{request.Height}x{requestedFrames}f -> {width}x{height}x{frames}f.");
        }

        MiniMaxH3GenerationRequest inner = new MiniMaxH3GenerationRequest
        {
            Width = width,
            Height = height,
            LatentFrames = MiniMaxH3Geometry.VideoLatentFrames(frames),
            AudioLatentFrames = MiniMaxH3Geometry.AudioLatentFrames(frames),
            Steps = request.Steps ?? 30,
            Seed = (int)(RecipeRequestMapper.MapSeed(request.Seed) ?? 0),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        try
        {
            MiniMaxH3TextEncoder.Result encoded = _textEncoder.Encode(_backend, _tokenizer, request.Prompt);
            MiniMaxH3Pipeline.Result result;
            try
            {
                result = _pipeline.Generate(encoded.HiddenStates, inner, encoded.TagRuns, bridge);
            }
            finally
            {
                encoded.HiddenStates.Dispose();
            }
            AudioBuffer audio = AudioBuffer.FromChannels(result.Audio, result.AudioSampleRate);
            Logs.Info($"[MiniMaxH3RecipePipeline] {result.Frames.Length} frames {result.Width}x{result.Height}"
                + (audio.IsEmpty ? "." : $" plus a {audio.SampleRate} Hz {audio.ChannelCount}ch soundtrack."));
            return VideoRecipeUtils.ToResult(result.Frames, result.Width, result.Height, request,
                audio.IsEmpty ? null : audio);
        }
        catch (Exception ex)
        {
            Logs.Error("[MiniMaxH3RecipePipeline] Generation failed.", ex);
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _textEncoder.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
