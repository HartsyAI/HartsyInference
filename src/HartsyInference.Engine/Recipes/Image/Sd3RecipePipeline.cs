using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SD3 pipeline driven against the native <see cref="ImageRequest"/>. Both CLIPs share one BPE tokenizer (encoded once, reused for L and G); when a T5 encoder is present the prompt is additionally tokenized with the T5 SentencePiece plus its attention mask. Runs <see cref="Sd3Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Sd3Loader.Generate</c> drive path (text-to-image only).</summary>
public sealed class Sd3RecipePipeline : IRecipePipeline
{
    private readonly Sd3Pipeline _pipeline;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly T5Tokenizer? _t5Tokenizer;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed SD3 pipeline plus its tokenizers, taking ownership of every disposable. <paramref name="loaders"/> holds the checkpoint plus one loader per component resolved as a separate file, and must outlive the weights it mmaps.</summary>
    public Sd3RecipePipeline(Sd3Pipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer? t5Tokenizer, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _pipeline = pipeline;
        _clipTokenizer = clipTokenizer;
        _t5Tokenizer = t5Tokenizer;
        _loaders = loaders;
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";

        // Both CLIPs share the same BPE tokenizer; encode once and reuse for L and G.
        int[] promptTokensClip = _clipTokenizer.Encode(prompt);
        int[] negTokensClip = _clipTokenizer.Encode(negative);
        int promptEos = ClipTokenizer.FindEosPosition(promptTokensClip);
        int negEos = ClipTokenizer.FindEosPosition(negTokensClip);

        int[]? promptTokensT5 = null;
        int[]? negTokensT5 = null;
        int[]? promptMaskT5 = null;
        int[]? negMaskT5 = null;
        if (_t5Tokenizer is not null)
        {
            promptTokensT5 = _t5Tokenizer.Encode(prompt);
            negTokensT5 = _t5Tokenizer.Encode(negative);
            promptMaskT5 = T5Tokenizer.CreateAttentionMask(promptTokensT5);
            negMaskT5 = T5Tokenizer.CreateAttentionMask(negTokensT5);
        }

        // Sd3Pipeline validates the source against the unsnapped request size, so resolve at exactly that.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            RecipeRequestMapper.ToTextToImage(request, negative) with
            {
                ClipSkip = RecipeRequestMapper.MapClipSkip(request.ClipSkip),
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokensClip, negTokensClip,
            promptTokensClip, negTokensClip,
            promptEos, negEos,
            promptEos, negEos,
            promptTokensT5, negTokensT5,
            promptMaskT5, negMaskT5,
            inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "sd3",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = (request.Steps ?? Sd3Recipe.FamilyDefaults.Steps).ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _clipTokenizer.Dispose();
        _t5Tokenizer?.Dispose();
        // The LoRA stack owns the merged tensors the transformer/CLIP encoders reference, so it outlives them
        // by exactly this much (same pattern as Flux1RecipePipeline/SdxlRecipePipeline).
        _loraStack?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
