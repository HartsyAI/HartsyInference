using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Lens pipeline driven against the native <see cref="ImageRequest"/>. Renders the prompt through Lens' Harmony chat template (<see cref="GptOssTokenizer.BuildChatInputs"/>) and calls <see cref="LensPipeline.GenerateFromTokens"/>, which owns the GPT-OSS-20B encoder forward. Mirrors the SwarmUI backend's <c>LensLoader.Generate</c> drive path. Wraps the factory-built Lens bundle plus its tokenizer, taking ownership of both.</summary>
public sealed class LensRecipePipeline(LensPipelineBundle bundle, LensConfig config, GptOssTokenizer tokenizer,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly LensPipelineBundle _bundle = bundle;
    private readonly LensConfig _config = config;
    private readonly GptOssTokenizer _tokenizer = tokenizer;
    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <summary>The loaded <see cref="LensConfig"/> already carries the variant's official step count and CFG (Turbo 4/1.0, standard 20/5.0), so the defaults are read straight off it.</summary>
    public ImageDefaults? VariantDefaults => new ImageDefaults { Steps = _config.DefaultSteps, CfgScale = _config.DefaultCfgScale, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;


        (int[] posTokens, _) = _tokenizer.BuildChatInputs(prompt);
        // Negative tokens only matter when CFG is live; Lens-Turbo (cfg 1) skips the second pass.
        int[]? negTokens = cfgScale > 1f ? _tokenizer.BuildChatInputs(negative).tokenIds : null;

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth / 16 * 16, reqHeight / 16 * 16);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
