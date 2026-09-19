using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SD3 pipeline driven against the native <see cref="ImageRequest"/>. Both CLIPs share one BPE tokenizer (encoded once, reused for L and G); when a T5 encoder is present the prompt is additionally tokenized with the T5 SentencePiece plus its attention mask. Runs <see cref="Sd3Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Sd3Loader.Generate</c> drive path (text-to-image only). Wraps the constructed SD3 pipeline plus its tokenizers, taking ownership of every disposable. <paramref name="loaders"/> holds the open checkpoint — either container, plus any copies the backend needed widened — and one loader per component resolved as a separate file, and must outlive the weights they map.</summary>
public sealed class Sd3RecipePipeline(Sd3Pipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer? t5Tokenizer,
    List<IDisposable> loaders, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly Sd3Pipeline _pipeline = pipeline;
    private readonly ClipTokenizer _clipTokenizer = clipTokenizer;
    private readonly T5Tokenizer? _t5Tokenizer = t5Tokenizer;
    private readonly List<IDisposable> _loaders = loaders;
    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";

        // Declaring ComfyBlend is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar
        // now. SD3 is the only wired family whose CLIP HIDDEN states reach the DiT, so all three arms are
        // blended rather than T5 alone.
        // Both CLIPs share the same BPE tokenizer; encode once and reuse for L and G.
        (int[] promptTokensClip, float[]? promptClipWeights) = TokenizeClipWeighted(prompt);
        (int[] negTokensClip, float[]? negClipWeights) = TokenizeClipWeighted(negative);
        int promptEos = ClipTokenizer.FindEosPosition(promptTokensClip);
        int negEos = ClipTokenizer.FindEosPosition(negTokensClip);

        int[]? promptTokensT5 = null;
        int[]? negTokensT5 = null;
        int[]? promptMaskT5 = null;
        int[]? negMaskT5 = null;
        float[]? promptT5Weights = null;
        float[]? negT5Weights = null;
        int[]? emptyT5 = null;
        int[]? emptyMaskT5 = null;
        if (_t5Tokenizer is not null)
        {
            (promptTokensT5, promptT5Weights) = T5WeightedConditioning.Tokenize(_t5Tokenizer, prompt);
            (negTokensT5, negT5Weights) = T5WeightedConditioning.Tokenize(_t5Tokenizer, negative);
            promptMaskT5 = T5Tokenizer.CreateAttentionMask(promptTokensT5);
            negMaskT5 = T5Tokenizer.CreateAttentionMask(negTokensT5);
            emptyT5 = T5WeightedConditioning.EmptyTokens(_t5Tokenizer);
            emptyMaskT5 = T5Tokenizer.CreateAttentionMask(emptyT5);
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

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokensClip, negTokensClip,
            promptTokensClip, negTokensClip,
            promptEos, negEos,
            promptEos, negEos,
            promptTokensT5, negTokensT5,
            promptMaskT5, negMaskT5,
            inner, bridge,
            promptClipWeights, negClipWeights, promptT5Weights, negT5Weights, emptyT5, emptyMaskT5);

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

    /// <summary>The CLIP tokenization plus its per-token weights. An unweighted prompt keeps
    /// <see cref="ClipTokenizer.Encode"/> verbatim, so its ids are exactly what they were before weighting
    /// existed.</summary>
    /// <remarks>Chunk 0 only, which is what the single-array pipeline signature can carry and what the plain
    /// encode already produced: a prompt past the 77-token window is truncated the same way either way. A
    /// multi-chunk weighted prompt (<c>&lt;break&gt;</c>) would need the chunked signature and is not wired.</remarks>
    private (int[] Tokens, float[]? Weights) TokenizeClipWeighted(string prompt)
    {
        string text = PromptTagFlattening.Flatten(prompt);
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(text);
        if (!PromptWeighting.HasWeights(spans))
        {
            return (_clipTokenizer.Encode(PromptWeighting.Join(spans)), null);
        }
        (IReadOnlyList<int[]> ids, IReadOnlyList<float[]> weights) =
            WeightedPromptTokenizer.Tokenize(_clipTokenizer, text);
        return (ids[0], weights[0]);
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
        foreach (IDisposable loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
