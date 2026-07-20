using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed F-Lite pipeline driven against the native <see cref="ImageRequest"/>. <see cref="FLitePipeline"/> owns the T5-XXL encoder; this tokenizes the prompt (no attention mask beyond an all-ones window — the fal-ai reference attends the full 512-token pad sea) and passes a null negative when unset (the reference uses a zero context, not an encoded empty string). Runs <see cref="FLitePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>FLiteLoader.Generate</c> drive path (text-to-image only).</summary>
public sealed class FLiteRecipePipeline : IRecipePipeline
{
    private readonly FLitePipeline _pipeline;
    private readonly T5Tokenizer _tokenizer;
    private readonly FLiteCheckpointConverter.LoaderHandle _handle;

    /// <summary>Wraps the constructed F-Lite pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public FLiteRecipePipeline(FLitePipeline pipeline, T5Tokenizer tokenizer, FLiteCheckpointConverter.LoaderHandle handle)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _handle = handle;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";

        // fal-ai reference: no real attention mask (all-ones window over the padded context), and a ZERO
        // context for an unset negative rather than an encoded empty string.
        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] promptMask = new int[promptTokens.Length];
        Array.Fill(promptMask, 1);
        int[]? negTokens = string.IsNullOrWhiteSpace(negative) ? null : _tokenizer.Encode(negative);
        int[]? negMask = negTokens is null ? null : promptMask;

        TextToImageRequest inner = RecipeRequestMapper.ToTextToImage(request, negative);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, promptMask, negTokens, negMask, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "f-lite",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = (request.Steps ?? FLiteRecipe.FamilyDefaults.Steps).ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _handle.Dispose();
    }
}
