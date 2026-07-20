using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Lens pipeline driven against the native <see cref="ImageRequest"/>. Renders the prompt through Lens' Harmony chat template (<see cref="GptOssTokenizer.BuildChatInputs"/>) and calls <see cref="LensPipeline.GenerateFromTokens"/>, which owns the GPT-OSS-20B encoder forward. Mirrors the SwarmUI backend's <c>LensLoader.Generate</c> drive path.</summary>
public sealed class LensRecipePipeline : IRecipePipeline
{
    private readonly LensPipelineBundle _bundle;
    private readonly LensConfig _config;
    private readonly GptOssTokenizer _tokenizer;

    /// <summary>Wraps the factory-built Lens bundle plus its tokenizer, taking ownership of both.</summary>
    public LensRecipePipeline(LensPipelineBundle bundle, LensConfig config, GptOssTokenizer tokenizer)
    {
        _bundle = bundle;
        _config = config;
        _tokenizer = tokenizer;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps > 0 ? request.Steps : _config.DefaultSteps;
        float cfgScale = request.CfgScale <= 0 ? _config.DefaultCfgScale : request.CfgScale;

        // TODO(E-IMG-4): img2img/inpaint (request.Img2Img/Inpaint) is not yet mapped — text-to-image only.

        (int[] posTokens, _) = _tokenizer.BuildChatInputs(prompt);
        // Negative tokens only matter when CFG is live; Lens-Turbo (cfg 1) skips the second pass.
        int[]? negTokens = cfgScale > 1f ? _tokenizer.BuildChatInputs(negative).tokenIds : null;

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _bundle.Pipeline.GenerateFromTokens(posTokens, negTokens, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "lens",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfgScale.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _bundle.Dispose();
        _tokenizer.Dispose();
    }
}
