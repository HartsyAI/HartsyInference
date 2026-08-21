using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Lance text-to-video pipeline driven against the native <see cref="VideoRequest"/>. Tokenizes
/// the caption only — the pipeline wraps it in the upstream chat-templated scaffold itself — and calls
/// <see cref="LanceVideoPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>LanceLoader.Generate</c>
/// text-to-video branch.</summary>
public sealed class LanceVideoRecipePipeline : IVideoRecipePipeline
{
    /// <summary>Total downscale between pixels and transformer tokens (VAE 16x, latent patch (1,1,1) per the real checkpoint).</summary>
    private const int SizeMultiple = 16;

    /// <summary>Lance tops out at 121 frames (~5s @ 24fps) on a 4x temporal VAE compression.</summary>
    private const int MaxFrames = 121;

    private readonly LanceVideoPipeline _pipeline;
    private readonly LanceConfig _config;
    private readonly LanceTransformer _transformer;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IReadOnlyList<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Lance video pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public LanceVideoRecipePipeline(LanceVideoPipeline pipeline, LanceConfig config, LanceTransformer transformer, ILlmTokenizer tokenizer,
        IReadOnlyList<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _config = config;
        _transformer = transformer;
        _tokenizer = tokenizer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumTimesteps;
        float cfgScale = request.CfgScale ?? _config.CfgTextScale;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, SizeMultiple);
        int numFrames = Math.Min(MaxFrames, VideoRecipeUtils.ResolveFrames(request, modelDefault: 81, step: 4));

        // An empty negative replicates the upstream uncond (caption dropped); a non-empty one substitutes it.
        int[] promptTokens = _tokenizer.EncodeOrdinary(prompt);
        int[] negativeTokens = string.IsNullOrWhiteSpace(negative) ? Array.Empty<int>() : _tokenizer.EncodeOrdinary(negative);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            // Carry the sampler selection into the inner request. LanceVideoPipeline drives the sampler seam off
            // TextToImageRequest.Scheduler; without this the engine path builds an inner request with it unset and
            // the user's choice is silently dropped — the failure the seam exists to remove. (SamplingParamResolver
            // takes an ImageRequest, so this is not a copy of the MageFlow line.)
            Scheduler = request.Sampler,
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        try
        {
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromTokens(promptTokens, negativeTokens, inner, numFrames, bridge);
            Logs.Info($"[LanceVideoRecipePipeline] T2V returned {frames.Length} frames {outW}x{outH}.");
            return VideoRecipeUtils.ToResult(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[LanceVideoRecipePipeline] Generation failed.", ex);
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _transformer.Dispose();
        (_tokenizer as IDisposable)?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
