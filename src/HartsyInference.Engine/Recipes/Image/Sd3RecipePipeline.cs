using System.Globalization;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SD3 pipeline driven against the native <see cref="ImageRequest"/>. Both CLIPs share one BPE tokenizer (encoded once, reused for L and G); when a T5 encoder is present the prompt is additionally tokenized with the T5 SentencePiece plus its attention mask. Runs <see cref="Sd3Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Sd3Loader.Generate</c> drive path (text-to-image only).</summary>
public sealed class Sd3RecipePipeline : IRecipePipeline
{
    private readonly Sd3Pipeline _pipeline;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly T5Tokenizer? _t5Tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader? _t5Loader;

    /// <summary>Wraps the constructed SD3 pipeline plus its tokenizers, taking ownership of every disposable (the T5 loader is null when T5 was bundled in the checkpoint).</summary>
    public Sd3RecipePipeline(Sd3Pipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer? t5Tokenizer, SafeTensorsLoader checkpointLoader, SafeTensorsLoader? t5Loader)
    {
        _pipeline = pipeline;
        _clipTokenizer = clipTokenizer;
        _t5Tokenizer = t5Tokenizer;
        _checkpointLoader = checkpointLoader;
        _t5Loader = t5Loader;
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

        // TODO(E-IMG-4/5): img2img / inpaint are not yet mapped — the SwarmUI loader built an
        // ImageToImageRequest here. Text-to-image only.
        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps <= 0 ? null : request.Steps,
            CfgScale = request.CfgScale <= 0 ? null : (float?)request.CfgScale,
            Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
            ClipSkip = request.ClipSkip <= 0 ? null : request.ClipSkip,
        };

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
                ["steps"] = request.Steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _clipTokenizer.Dispose();
        _t5Tokenizer?.Dispose();
        _checkpointLoader.Dispose();
        _t5Loader?.Dispose();
    }
}
