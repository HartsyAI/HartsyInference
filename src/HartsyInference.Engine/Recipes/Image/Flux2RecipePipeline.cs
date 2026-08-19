using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

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
    private readonly IDisposable? _ggufHandle;

    /// <summary>Wraps the constructed Flux.2 pipeline plus its tokenizer, taking ownership of every disposable. Exactly one of <paramref name="qwenTokenizer"/> (Klein) / <paramref name="mistralTokenizer"/> (Dev) is non-null. <paramref name="ggufHandle"/> is non-null when the transformer loaded from a GGUF file (keeps the mmap alive for any pass-through F16 tensor still referencing it).</summary>
    private readonly MergedLoraStack? _loraStack;

    public Flux2RecipePipeline(Flux2Pipeline pipeline, Flux2Config config, Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer, string mistralSystemPrompt, LlamaStyleEncoder encoder, List<SafeTensorsLoader> loaders, IDisposable? ggufHandle = null, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _config = config;
        _qwenTokenizer = qwenTokenizer;
        _mistralTokenizer = mistralTokenizer;
        _mistralSystemPrompt = mistralSystemPrompt;
        _encoder = encoder;
        _loaders = loaders;
        _ggufHandle = ggufHandle;
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

        // Resolved at the 16-rounded size Flux2Pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        RegionalPlan? regionalPlan = null;
        try
        {
            regionalPlan = BuildRegionalPlan(prompt, width, height, steps);

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
                tokenIds, inner, guidanceScale: guidance, onProgress: bridge, regionalPlan: regionalPlan);

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
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
        }
    }

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c>
    /// parts, null otherwise (Tier 3.7). Mirrors <c>Flux1RecipePipeline.BuildRegionalPlan</c>: each region's text is
    /// tokenized the SAME way the base prompt was (Klein's chat template vs. Dev's Mistral splice) and encoded
    /// through <see cref="Flux2Pipeline.EncodeRegionText"/> — the same text-encoder instance + hidden-layer taps
    /// the base prompt uses. <see cref="RegionalPlan.BaseCond"/> is a required field on the resolver's signature
    /// that <see cref="Flux2Pipeline.GenerateFromTokens"/>'s regional path never reads (same as Flux.1 — confirmed
    /// by inspection: the pipeline builds its own background stream from the base <c>textEmbeddings</c>) — a
    /// throwaway placeholder tensor satisfies it.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            int[] regionTokenIds = _config.TextEncoderType == Flux2TextEncoderType.Mistral
                ? BuildMistralDevTokenIds(_mistralTokenizer!, text)
                : _qwenTokenizer!.EncodeChat(text);
            return _pipeline.EncodeRegionText(regionTokenIds);
        });
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
        _ggufHandle?.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
