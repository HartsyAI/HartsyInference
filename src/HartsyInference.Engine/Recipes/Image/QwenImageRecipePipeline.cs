using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Qwen-Image pipeline driven against the native <see cref="ImageRequest"/>. <see cref="QwenImagePipeline"/> owns the text-encoder forward, so this only builds the templated Qwen token ids plus the prefix-drop indices and calls <see cref="QwenImagePipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>QwenImageLoader.Generate</c> text-to-image drive path.</summary>
public sealed class QwenImageRecipePipeline : IRecipePipeline
{
    /// <summary>The exact system prompt Qwen-Image conditions on (diffusers <c>QwenImagePipeline.prompt_template_encode</c>); its hidden states are dropped by the prefix-drop index.</summary>
    private const string QwenImageSystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    /// <summary>diffusers truncates the templated sequence at 512 tokens.</summary>
    private const int MaxTokens = 512;

    /// <summary>Qwen-Image's guidance default when the caller does not supply one (SwarmUI loader fallback).</summary>
    private const float DefaultCfg = 2.5f;

    private readonly QwenImagePipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly QwenImageTransformer _transformer;
    private readonly QwenImageVaeDecoder _vae;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly IDisposable? _ggufHandle;

    /// <summary>Wraps the constructed Qwen-Image pipeline plus its components, taking ownership of every disposable.</summary>
    public QwenImageRecipePipeline(QwenImagePipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, QwenImageTransformer transformer, QwenImageVaeDecoder vae, QwenImageVaeEncoder? vaeEncoder, List<SafeTensorsLoader> loaders, IDisposable? ggufHandle)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vae = vae;
        _vaeEncoder = vaeEncoder;
        _loaders = loaders;
        _ggufHandle = ggufHandle;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? QwenImageRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? QwenImageRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4/5): Qwen-Image-Edit reference images (editRefImages / editRefVisionImages /
        // editRefTimestepZero), LoRA, ControlNet and regional prompting are deferred — the SwarmUI loader resolved
        // those from T2IParamInput too. Classic strength-based img2img/inpaint is wired below; the edit path is a
        // distinct conditioning mode and stays deferred.
        (int[] promptTokens, int promptDrop) = EncodeWithTemplate(_tokenizer, prompt);
        (int[] negTokens, int negDrop) = EncodeWithTemplate(_tokenizer, negative);

        // Qwen-Image is the only family offering BOTH modes, so the choice is explicit rather than inferred:
        // Reference routes the init image to the in-context edit tokens, anything else to the classic noised start.
        bool refEdit = request.Img2Img?.Mode == Img2ImgMode.Reference;
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? initImage = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        Img2ImgResolver.Img2ImgSpec? img2img = refEdit ? null : initImage;
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, negTokens, inner, bridge,
            promptDropIndex: promptDrop, negativeDropIndex: negDrop,
            editRefImages: refEdit && initImage is not null ? [initImage.SourceTensor] : null);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "qwen-image",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Builds the Qwen-Image templated token sequence (real length, no padding — the pipeline has no attention mask) plus the prefix-drop index: the count of leading system-block + user-header tokens whose hidden states the pipeline discards (diffusers' <c>prompt_template_encode_start_idx</c>). Special tokens are inserted by id; the text between them is BPE'd per segment.</summary>
    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new List<int>(64);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(QwenImageSystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n"));
        int dropIndex = ids.Count;
        ids.AddRange(tokenizer.EncodeRaw(prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        if (ids.Count > MaxTokens)
        {
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        }
        return (ids.ToArray(), dropIndex);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        _vae.Dispose();
        _vaeEncoder?.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        _ggufHandle?.Dispose();
    }
}
