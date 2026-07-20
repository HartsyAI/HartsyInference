using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Flux.2 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Flux2Pipeline"/> owns the text encoder, so this only produces the token ids — the embedded Qwen3 chat template for Klein, or the spliced Mistral tekken conditioning ids for Dev — and calls <see cref="Flux2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Flux2Loader.Generate</c> text-to-image drive path.</summary>
public sealed class Flux2RecipePipeline : IRecipePipeline
{
    private readonly Flux2Pipeline _pipeline;
    private readonly Flux2Config _config;
    private readonly Qwen3Tokenizer? _qwenTokenizer;
    private readonly ErnieTokenizer? _mistralTokenizer;
    private readonly string _mistralSystemPrompt;
    private readonly LlamaStyleEncoder _encoder;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Flux.2 pipeline plus its tokenizer, taking ownership of every disposable. Exactly one of <paramref name="qwenTokenizer"/> (Klein) / <paramref name="mistralTokenizer"/> (Dev) is non-null.</summary>
    public Flux2RecipePipeline(Flux2Pipeline pipeline, Flux2Config config, Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer, string mistralSystemPrompt, LlamaStyleEncoder encoder, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _config = config;
        _qwenTokenizer = qwenTokenizer;
        _mistralTokenizer = mistralTokenizer;
        _mistralSystemPrompt = mistralSystemPrompt;
        _encoder = encoder;
        _loaders = loaders;
    }

    /// <summary>A Klein checkpoint (no guidance embedding) is CFG-distilled and few-step, so it resolves against <see cref="Flux2Recipe.KleinDefaults"/> rather than Dev's 50 steps.</summary>
    public ImageDefaults? VariantDefaults => _config.GuidanceEmbed ? Flux2Recipe.FamilyDefaults : Flux2Recipe.KleinDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        int steps = request.Steps ?? (_config.GuidanceEmbed ? Flux2Recipe.FamilyDefaults.Steps : Flux2Recipe.KleinDefaults.Steps);
        // Flux.2 rounds image dims down to a multiple of 16 (VAE 8× × 2×2 patch).
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = (reqWidth / 16) * 16;
        int height = (reqHeight / 16) * 16;
        // Klein has no guidance embedding; Dev uses guidance ~3.5 (BFL distillation target).
        float guidance = _config.GuidanceEmbed ? 3.5f : 0f;

        // TODO(E-IMG-4/5): img2img, NegativePrompt/CfgScale mapping, and user component overrides are deferred.
        int[] tokenIds = _config.TextEncoderType == Flux2TextEncoderType.Mistral
            ? BuildMistralDevTokenIds(_mistralTokenizer!, prompt)
            : _qwenTokenizer!.EncodeChat(prompt);

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            Width = width,
            Height = height,
            Steps = steps,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            tokenIds, inner, guidanceScale: guidance, onProgress: bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "flux2",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Builds Flux.2 Dev conditioning ids: <c>&lt;s&gt;[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]prompt[/INST]</c>. Special markers are spliced as raw ids (BOS=1, [SYSTEM_PROMPT]=17, [/SYSTEM_PROMPT]=18, [INST]=3, [/INST]=4) around byte-level BPE segments — special strings are pre-token boundaries in the HF reference, so segment-wise encoding is id-exact. No EOS (ComfyUI <c>has_end_token=False</c>).</summary>
    private int[] BuildMistralDevTokenIds(ErnieTokenizer tokenizer, string prompt)
    {
        List<int> ids = new List<int>(256) { 1, 17 };
        ids.AddRange(tokenizer.EncodeRaw(ByteLevelCodec.Encode(_mistralSystemPrompt)));
        ids.Add(18);
        ids.Add(3);
        ids.AddRange(tokenizer.EncodeRaw(ByteLevelCodec.Encode(prompt)));
        ids.Add(4);
        return ids.ToArray();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwenTokenizer?.Dispose();
        _mistralTokenizer?.Dispose();
        _encoder.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
