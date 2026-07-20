using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Chroma pipeline driven against the native <see cref="ImageRequest"/>. <see cref="ChromaPipeline"/> owns the T5-XXL encoder, so this only tokenizes the prompt/negative (plus the tokenizer attention masks Chroma's "first padding token unmasked" rule needs) and calls <see cref="ChromaPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>ChromaLoader.Generate</c> drive path.</summary>
public sealed class ChromaRecipePipeline : IRecipePipeline
{
    private readonly ChromaPipeline _pipeline;
    private readonly T5Tokenizer _tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _t5Loader;
    private readonly SafeTensorsLoader _vaeLoader;

    /// <summary>Wraps the constructed Chroma pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public ChromaRecipePipeline(ChromaPipeline pipeline, T5Tokenizer tokenizer, SafeTensorsLoader checkpointLoader, SafeTensorsLoader t5Loader, SafeTensorsLoader vaeLoader)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _checkpointLoader = checkpointLoader;
        _t5Loader = t5Loader;
        _vaeLoader = vaeLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps;
        // <=0 defers to the pipeline's ChromaConfig default (5.0); a positive request value overrides it.
        float? cfg = request.CfgScale <= 0 ? null : request.CfgScale;

        // TODO(E-IMG-4): img2img/inpaint (request.Img2Img/Inpaint) not yet mapped — text-to-image only.

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = steps,
            CfgScale = cfg,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
        };

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
                ["arch"] = "chroma",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
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
        _vaeLoader.Dispose();
    }
}
