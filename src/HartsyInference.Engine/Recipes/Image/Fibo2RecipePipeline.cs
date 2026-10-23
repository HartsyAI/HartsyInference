using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Fibo 2.0 pipeline driven by a native <see cref="ImageRequest"/>. The caption and the null-caption branch are
/// tokenized here through <see cref="Fibo2PromptTemplate"/>; the pipeline encodes and samples. Owns every disposable it was given, and
/// disposes them in reverse construction order.</summary>
public sealed partial class Fibo2RecipePipeline : IRecipePipeline
{
    private readonly Fibo2Pipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly ImageDefaults _defaults;
    private readonly ResolvedModelVariant _variant;
    private readonly IReadOnlyList<IDisposable> _owned;
    private int _disposed;

    /// <summary>Wraps the constructed pipeline. <paramref name="owned"/> is every disposable the recipe built, in construction order.</summary>
    public Fibo2RecipePipeline(Fibo2Pipeline pipeline, Qwen3Tokenizer tokenizer, ImageDefaults defaults, ResolvedModelVariant variant,
        IReadOnlyList<IDisposable> owned)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(defaults);
        ArgumentNullException.ThrowIfNull(owned);
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _defaults = defaults;
        _variant = variant;
        _owned = owned;
    }

    /// <inheritdoc/>
    public ImageDefaults? VariantDefaults => _defaults;

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">The request asks for something Fibo 2.0 does not implement: a negative prompt, reference
    /// images, or a sampler or schedule other than the built-in flow-matching Euler.</exception>
    public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(settings);
        cancel.ThrowIfCancellationRequested();
        if (IsEdit(request))
            return GenerateEdit(request, settings, progress, cancel);
        if (!string.IsNullOrWhiteSpace(request.NegativePrompt))
            throw new NotSupportedException("Fibo 2.0 does not map negative prompts yet: its unconditional branch is the fixed null caption.");
        if (!IsDefaultSampler(request.Sampler) || !IsDefaultSchedule(request.Scheduler))
            throw new NotSupportedException("Fibo 2.0 samples only with its built-in flow-matching Euler; drop the sampler and schedule selection.");

        int[] conditional = Fibo2PromptTemplate.Tokenize(_tokenizer, Fibo2PromptTemplate.CaptionFor(request.Prompt));
        int[] unconditional = Fibo2PromptTemplate.Tokenize(_tokenizer, Fibo2PromptTemplate.NullCaption);

        // Rounded down to the 16-pixel grid (VAE 8× and the denoiser's 2×2 patch), as the Flux.2 recipe does.
        (int reqWidth, int reqHeight) = (settings.Width, settings.Height);
        int width = reqWidth / 16 * 16;
        int height = reqHeight / 16 * 16;
        int steps = settings.Steps;
        float guidance = settings.CfgScale;
        float shift = (float)(request.SigmaShift ?? _defaults.SigmaShift ?? Math.Exp(1.15));
        int? seed = RecipeRequestMapper.MapSeed(request.Seed);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel, totalSteps: steps);
        (byte[] rgb, int outWidth, int outHeight, int usedSeed) = _pipeline.GenerateFromTokens(conditional, unconditional, width, height,
            steps, guidance, shift, seed, bridge);
        return new ImageResult
        {
            Rgb = rgb,
            Width = outWidth,
            Height = outHeight,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "fibo2",
                ["variant"] = _variant.Variant.Id,
                ["size"] = string.Create(CultureInfo.InvariantCulture, $"{outWidth}x{outHeight}"),
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Euler is the built-in sampler; hosts send it as their default, so it is accepted rather than refused.</summary>
    private static bool IsDefaultSampler(string? sampler) =>
        string.IsNullOrWhiteSpace(sampler) || string.Equals(sampler.Trim(), "euler", StringComparison.OrdinalIgnoreCase);

    /// <summary>The identity schedule names are accepted; any other sigma schedule is refused.</summary>
    private static bool IsDefaultSchedule(string? schedule) =>
        string.IsNullOrWhiteSpace(schedule) || schedule.Trim().ToLowerInvariant() is "normal" or "default";

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        for (int i = _owned.Count - 1; i >= 0; i--)
            _owned[i].Dispose();
    }
}
