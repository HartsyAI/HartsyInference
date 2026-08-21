using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

/// <summary>A constructed Lance image pipeline driven against the native <see cref="ImageRequest"/>. Tokenizes the caption only — the pipeline wraps it in the upstream chat-templated scaffold itself — and calls <see cref="LanceImagePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>LanceLoader.Generate</c> text-to-image drive path.</summary>
public sealed class LanceImageRecipePipeline : IRecipePipeline
{
    /// <summary>Total downscale between pixels and transformer tokens (VAE 16x, latent patch (1,1,1) per the real checkpoint).</summary>
    private const int SizeMultiple = 16;

    private readonly LanceImagePipeline _pipeline;
    private readonly LanceConfig _config;
    private readonly LanceTransformer _transformer;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IReadOnlyList<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Lance pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public LanceImageRecipePipeline(LanceImagePipeline pipeline, LanceConfig config, LanceTransformer transformer, ILlmTokenizer tokenizer, IReadOnlyList<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _config = config;
        _transformer = transformer;
        _tokenizer = tokenizer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumTimesteps;
        float cfgScale = request.CfgScale ?? _config.CfgTextScale;
        (int width, int height) = RecipeRequestMapper.Size(request);
        if (width % SizeMultiple != 0 || height % SizeMultiple != 0)
        {
            int fixedW = Math.Max(SizeMultiple, width / SizeMultiple * SizeMultiple);
            int fixedH = Math.Max(SizeMultiple, height / SizeMultiple * SizeMultiple);
            throw new InvalidOperationException(
                $"Lance requires width/height divisible by {SizeMultiple}. Got {width}x{height}; nearest valid is {fixedW}x{fixedH}.");
        }

        // TODO(E-IMG-4): img2img (request.Img2Img) needs a Wan22VaeEncoder on construction; masked inpaint is
        // unsupported upstream. Text-to-image only.

        // An empty negative replicates the upstream uncond (caption dropped); a non-empty one substitutes it.
        int[] promptTokens = _tokenizer.EncodeOrdinary(prompt);
        int[] negativeTokens = string.IsNullOrWhiteSpace(negative) ? Array.Empty<int>() : _tokenizer.EncodeOrdinary(negative);

        // Img2img only — a mask is refused upstream because Supports omits Inpaint.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = width,
                Height = height,
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

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(promptTokens, negativeTokens, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "lance-image",
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
