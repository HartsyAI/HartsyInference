using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed AuraFlow pipeline driven against the native <see cref="ImageRequest"/>. <see cref="AuraFlowPipeline"/> owns the Pile-T5-XL encoder, so this only tokenizes the prompt/negative (plus their T5 attention masks) and calls <see cref="AuraFlowPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>AuraFlowLoader.Generate</c> drive path (text-to-image only).</summary>
public sealed class AuraFlowRecipePipeline : IRecipePipeline
{
    private readonly AuraFlowPipeline _pipeline;
    private readonly T5Tokenizer _tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed AuraFlow pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public AuraFlowRecipePipeline(AuraFlowPipeline pipeline, T5Tokenizer tokenizer, SafeTensorsLoader checkpointLoader, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _checkpointLoader = checkpointLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(RecipeRequestMapper.ToTextToImage(request, negative), img2img);

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
                ["arch"] = "auraflow",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = (request.Steps ?? AuraFlowRecipe.FamilyDefaults.Steps).ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _checkpointLoader.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
