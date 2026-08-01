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

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Wan VACE pipeline driven against the native <see cref="VideoRequest"/>. The control clip
/// comes from <see cref="VideoRequest.InitImage"/> (tiled to the frame count — the extension's still-image control
/// branch), the prompt pair is umT5-encoded with the pad rows zeroed, and the whole thing goes through
/// <see cref="WanVacePipeline.GenerateFromControl"/>. Mirrors the SwarmUI backend's <c>WanVaceLoader.Generate</c>.</summary>
public sealed class WanVaceRecipePipeline : IVideoRecipePipeline
{
    /// <summary>Uniform control-hint scale passed to every VACE layer (ComfyUI hard-codes 1.0).</summary>
    private const float ControlScale = 1.0f;

    private readonly IBackend _backend;
    private readonly WanVacePipeline _pipeline;
    private readonly WanVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _umt5;
    private readonly WanVaceTransformer _transformer;
    private readonly IWanVaeEncoder _vaeEncoder;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed VACE pipeline plus its encoders, taking ownership of every disposable.</summary>
    public WanVaceRecipePipeline(IBackend backend, WanVacePipeline pipeline, WanVideoConfig config, T5Tokenizer tokenizer,
        T5TextEncoder umt5, WanVaceTransformer transformer, IWanVaeEncoder vaeEncoder, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _umt5 = umt5;
        _transformer = transformer;
        _vaeEncoder = vaeEncoder;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        // TODO(E-IMG-4/5): a real control VIDEO (and the reactive/inactive control mask) needs the host-side video
        // decoder the extension's ControlVideoDecoder wrapped; the native contract carries a single frame, so the
        // still is tiled across the clip (the extension's own still branch).
        ImageData control = request.InitImage
            ?? throw new InvalidOperationException(
                "Wan VACE needs a control image/video in VideoRequest.InitImage — that's the pose/depth/edge/sketch sequence the generation follows.");

        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: _config.VaeTemporalCompression);
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;
        (int width, int height) = VideoRecipeUtils.ResolveI2VResolution(request, control.Width, control.Height, _config.VaeSpatialCompression);

        byte[] controlRgb = VideoRecipeUtils.ResizeRgb24(control, width, height);
        Tensor controlClip = VideoRecipeUtils.TileRgbToClip(controlRgb, width, height, numFrames);

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
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromControl(promptEmbeds, negEmbeds, controlClip, inner, ControlScale, bridge);
            Logs.Info($"[WanVaceRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH} ({numFrames}f control).");
            return VideoRecipeUtils.ToResult(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVaceRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            controlClip.Dispose();
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

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
