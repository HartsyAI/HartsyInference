using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SDXL pipeline driven against the native <see cref="ImageRequest"/>. Encodes the prompt/negative
/// through the embedded CLIP-L/CLIP-G vocab and runs <see cref="SdxlPipeline.GenerateFromTokens"/>. Mirrors the proven
/// CLI image path so behavior is unchanged while the seam takes over.</summary>
public sealed class SdxlRecipePipeline : IRecipePipeline
{
    private readonly SdxlPipeline _pipeline;
    private readonly ClipTokenizer _tokenizer = new ClipTokenizer();

    /// <summary>Wraps the constructed SDXL pipeline, taking ownership.</summary>
    public SdxlRecipePipeline(SdxlPipeline pipeline) => _pipeline = pipeline;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string negative = request.NegativePrompt ?? "";

        int[] tokensL = _tokenizer.Encode(request.Prompt);
        int[] negL = _tokenizer.Encode(negative);
        int[] tokensG = _tokenizer.Encode(request.Prompt);
        int[] negG = _tokenizer.Encode(negative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = request.Prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps,
            CfgScale = request.CfgScale,
            Seed = request.Seed < 0 ? null : (int)request.Seed,
        };

        int totalSteps = request.Steps;
        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            tokensL, negL, tokensG, negG, eosG, negEosG, inner,
            p => progress?.Report(new StepPreview { Step = p.Step, TotalSteps = totalSteps }));

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "sdxl",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = request.Steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose() => _pipeline.Dispose();
}
