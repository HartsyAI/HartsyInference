using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SD1.5 pipeline driven against the native <see cref="ImageRequest"/>. Encodes the prompt/negative through the embedded CLIP-L vocab and runs <see cref="StableDiffusion15Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Sd15Loader.Generate</c> drive path (text-to-image only).</summary>
public sealed class Sd15RecipePipeline : IRecipePipeline
{
    private readonly StableDiffusion15Pipeline _pipeline;
    private readonly ClipTokenizer _tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;

    /// <summary>Wraps the constructed SD1.5 pipeline plus its tokenizer, taking ownership of every disposable.</summary>
    public Sd15RecipePipeline(StableDiffusion15Pipeline pipeline, ClipTokenizer tokenizer, SafeTensorsLoader checkpointLoader)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _checkpointLoader = checkpointLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string negative = request.NegativePrompt ?? "";

        int[] promptTokens = _tokenizer.Encode(request.Prompt);
        int[] negativeTokens = _tokenizer.Encode(negative);

        // TODO(E-IMG-4/5): img2img / inpaint / LoRA / ControlNet / IP-Adapter / variation-seed / weighted
        // conditioning are not yet mapped — the SwarmUI loader resolved all of these here. Text-to-image only.
        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = request.Prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps <= 0 ? null : request.Steps,
            CfgScale = request.CfgScale <= 0 ? null : (float?)request.CfgScale,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
            Scheduler = request.Scheduler,
            ClipSkip = request.ClipSkip <= 0 ? null : request.ClipSkip,
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, negativeTokens, inner, bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "sd15",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = request.Steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _checkpointLoader.Dispose();
    }
}
