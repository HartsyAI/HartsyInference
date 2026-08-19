using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Chroma Radiance pipeline driven against the native <see cref="ImageRequest"/>. <see cref="ChromaRadiancePipeline"/> owns the T5-XXL encoder, so this only tokenizes the prompt/negative (plus the tokenizer attention masks the "first padding token unmasked" rule needs) and calls <see cref="ChromaRadiancePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>ChromaRadianceLoader.Generate</c> drive path; the decode is pixel-space, so no VAE is involved.</summary>
public sealed class ChromaRadianceRecipePipeline : IRecipePipeline
{
    private readonly ChromaRadiancePipeline _pipeline;
    private readonly ChromaRadianceConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _t5Loader;

    /// <summary>Wraps the constructed Chroma Radiance pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public ChromaRadianceRecipePipeline(ChromaRadiancePipeline pipeline, ChromaRadianceConfig config, T5Tokenizer tokenizer, SafeTensorsLoader checkpointLoader, SafeTensorsLoader t5Loader)
    {
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _checkpointLoader = checkpointLoader;
        _t5Loader = t5Loader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.DefaultSteps;
        float cfgScale = request.CfgScale ?? _config.DefaultCfgScale;

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

        // Radiance is pixel-space, so no VAE encoder is involved — the source pixels are the clean sample. The
        // pipeline validates against the unpadded request size and pads source and mask alongside the latent itself.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
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
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, negTokens, promptMask, negMask, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "chroma-radiance",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfgScale.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _checkpointLoader.Dispose();
        _t5Loader.Dispose();
    }
}
