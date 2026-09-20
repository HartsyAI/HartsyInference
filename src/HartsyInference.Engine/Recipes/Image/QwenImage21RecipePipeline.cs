using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Qwen-Image 2.1 pipeline driven against the native <see cref="ImageRequest"/>. Builds the
/// chat-templated token ids and the system-turn drop index, then calls
/// <see cref="QwenImage21Pipeline.GenerateFromTokens"/>; the pipeline owns the encoder forward. Takes ownership of
/// every disposable it is handed.</summary>
public sealed class QwenImage21RecipePipeline(QwenImage21Pipeline pipeline, Qwen3Tokenizer tokenizer,
    LlamaStyleEncoder textEncoder, QwenImage21Transformer transformer, Wan22VaeDecoder vae,
    List<SafeTensorsLoader> loaders, IDisposable? checkpoint) : IRecipePipeline
{
    private readonly QwenImage21Pipeline _pipeline = pipeline;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly LlamaStyleEncoder _textEncoder = textEncoder;
    private readonly QwenImage21Transformer _transformer = transformer;
    private readonly Wan22VaeDecoder _vae = vae;
    private readonly List<SafeTensorsLoader> _loaders = loaders;
    private readonly IDisposable? _checkpoint = checkpoint;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int steps = request.Steps ?? QwenImage21Recipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? QwenImage21Recipe.FamilyDefaults.CfgScale;
        // 16x VAE and no patchify, so the image dims must be a multiple of 16. ComfyUI's own template rounds
        // reference and output sizes to 32; 16 is this pipeline's hard floor and what it snaps to.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = Math.Max(QwenImage21Pipeline.VaeScale, reqWidth / QwenImage21Pipeline.VaeScale * QwenImage21Pipeline.VaeScale);
        int height = Math.Max(QwenImage21Pipeline.VaeScale, reqHeight / QwenImage21Pipeline.VaeScale * QwenImage21Pipeline.VaeScale);

        (int[] condIds, float[]? condWeights) = Tokenize(_tokenizer, request.Prompt);
        int drop = _tokenizer.SystemTurnLength(QwenImage21Recipe.SystemPrompt);
        bool useCfg = cfg > 1.0f;
        int[]? uncondIds = null;
        float[]? uncondWeights = null;
        if (useCfg)
        {
            (uncondIds, uncondWeights) = Tokenize(_tokenizer, request.NegativePrompt ?? "");
        }

        Action<Diffusion.Requests.GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);
        int seed = RecipeRequestMapper.MapSeed(request.Seed) ?? Random.Shared.Next(int.MaxValue);

        using Tensor image = _pipeline.GenerateFromTokens(
            condIds, drop, uncondIds, drop, width, height, steps, cfg, seed,
            seamlessTiling: request.SeamlessTiling,
            variationSeed: request.VariationSeed?.Seed ?? -1,
            variationSeedStrength: request.VariationSeed?.Strength ?? 0,
            samplerSelection: SamplingParamResolver.ResolveSchedulerName(request),
            onProgress: bridge,
            condWeights: condWeights is null ? null : new WeightedTokenSequence(condIds, condWeights),
            uncondWeights: uncondWeights is null || uncondIds is null ? null : new WeightedTokenSequence(uncondIds, uncondWeights));

        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
        // Channel 3 is the model's own alpha, not a matte: Qwen-Image 2.1's VAE decodes RGBA and the prompt wording
        // ("This is an RGBA format image with transparency...") is how the user asks for it. Left null when the
        // image came out fully opaque, which is what ImageResult.Alpha's contract means by "no alpha".
        byte[]? alpha = ImagePostProcessor.TensorToChannelBytes(image, channel: 3);
        if (alpha is not null && IsEffectivelyOpaque(alpha))
        {
            alpha = null;
        }

        return new ImageResult
        {
            Rgb = rgb,
            Alpha = alpha,
            Width = width,
            Height = height,
            Seed = seed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "qwen-image-2.1",
                ["size"] = $"{width}x{height}",
                ["seed"] = seed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                ["alpha"] = alpha is null ? "opaque" : "rgba",
            },
        };
    }

    /// <summary>Chat-templated encode plus the per-token weights parsed from the emphasis grammar. The weights are
    /// returned at the sequence's full length, not shortened by the system-turn drop: the pipeline slices
    /// <c>drop</c> rows off the conditioning and <see cref="CondTokenWeights"/> right-aligns, so the offset is
    /// <c>−drop</c> and weight <c>i</c> lands on sequence position <c>i</c>. Shortening them here instead would
    /// shift every emphasis onto a later word.</summary>
    internal static (int[] Tokens, float[]? Weights) Tokenize(Qwen3Tokenizer tokenizer, string? prompt)
    {
        (int[] prefix, int[] suffix) = tokenizer.ChatTemplateIds(
            includeThinkBlock: false, systemPrompt: QwenImage21Recipe.SystemPrompt);
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build(
            PromptTagFlattening.Flatten(prompt),
            t => tokenizer.EncodeChatUnpadded(t, QwenImage21Recipe.SystemPrompt, includeThinkBlock: false),
            tokenizer.EncodeRaw,
            prefix,
            suffix);
        return (sequence.Tokens, sequence.IsUniformlyUnweighted ? null : sequence.Weights);
    }

    /// <summary>Alpha below which a pixel counts as genuinely transparent. The VAE reconstructs an opaque image's
    /// alpha at 253–255 rather than exactly 255 (measured on a plain "red apple" prompt: min 253, 32% of pixels
    /// under 255), so an exact test would attach a near-opaque plane to every ordinary generation and turn every
    /// PNG RGBA for nothing. A prompt that actually asks for transparency drives the background to ~0, far under
    /// this.</summary>
    private const byte OpaqueFloor = 250;

    private static bool IsEffectivelyOpaque(byte[] alpha)
    {
        foreach (byte a in alpha)
        {
            if (a < OpaqueFloor) return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        _checkpoint?.Dispose();
    }
}
