using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LTX-2.3 pipeline driven against the native <see cref="VideoRequest"/>. Tokenizes the prompt
/// pair with the Gemma SentencePiece tokenizer — the pipeline runs the text tower and the per-modality connectors
/// itself — and calls <see cref="LtxVideo2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's
/// <c>LtxVideo2Loader.Generate</c>.</summary>
public sealed class LtxVideo2RecipePipeline : IVideoRecipePipeline
{
    private readonly LtxVideo2Pipeline _pipeline;
    private readonly LtxVideo2Config _config;
    private readonly GemmaTokenizer _tokenizer;
    private readonly LlamaStyleEncoder _gemma;
    private readonly LtxVideo2Transformer _transformer;
    private readonly LtxVideo2TextConnectors _connectors;
    private readonly LtxAudioVocoder? _vocoder;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed LTX-2 pipeline plus its text tower, taking ownership of every disposable.</summary>
    public LtxVideo2RecipePipeline(LtxVideo2Pipeline pipeline, LtxVideo2Config config, GemmaTokenizer tokenizer, LlamaStyleEncoder gemma,
        LtxVideo2Transformer transformer, LtxVideo2TextConnectors connectors, LtxAudioVocoder? vocoder, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _gemma = gemma;
        _transformer = transformer;
        _connectors = connectors;
        _vocoder = vocoder;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 121, step: _config.VaeTemporalCompression);
        int frameRate = request.Fps ?? 24;
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        try
        {
            LtxVideo2Pipeline.Ltx2Result result = _pipeline.GenerateFromTokens(promptTokens, negTokens, inner, numFrames, frameRate, bridge);
            // TODO(E-IMG-4/5): the generated soundtrack (result.Audio) has no home on the frame-only video contract —
            // the extension muxed it into the container. Dropped here rather than silently mis-timed.
            if (result.Audio is not null && result.Audio.Length > 0)
            {
                Logs.Warning($"[LtxVideo2RecipePipeline] LTX-2 generated a {result.AudioSampleRate} Hz soundtrack, which the frame-only video contract cannot carry — dropped.");
            }
            Logs.Info($"[LtxVideo2RecipePipeline] Pipeline returned {result.Frames.Length} frames {result.Width}x{result.Height}.");
            return VideoRecipeUtils.ToVideoFrames(result.Frames, result.Width, result.Height, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[LtxVideo2RecipePipeline] Generation failed.", ex);
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _gemma.Dispose();
        _transformer.Dispose();
        _connectors.Dispose();
        (_vocoder as IDisposable)?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
