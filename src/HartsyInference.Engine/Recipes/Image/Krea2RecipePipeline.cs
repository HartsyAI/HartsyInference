using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Krea 2 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Krea2Pipeline"/> owns the Qwen3-VL-4B forward (it taps 12 decoder layers itself), so this only builds the templated token ids — byte-identical to Qwen-Image's template — plus the prefix-drop indices and calls <see cref="Krea2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Krea2Loader.Generate</c>.</summary>
public sealed class Krea2RecipePipeline : IRecipePipeline
{
    /// <summary>Krea 2's prompt template is byte-identical to Qwen-Image's — same system prompt, same prefix-drop design.</summary>
    private const string Krea2SystemPrompt =
        "system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, " +
        "spatial relationships of the objects and background:";

    /// <summary>The templated sequence is truncated at 512 tokens.</summary>
    private const int MaxTokens = 512;

    private readonly Krea2Pipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Krea2Transformer _transformer;
    private readonly QwenImageVaeDecoder _vae;
    private readonly QwenImageVaeEncoder? _vaeEncoder;
    private readonly bool _isTurbo;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Krea 2 pipeline plus its components, taking ownership of every disposable.</summary>
    public Krea2RecipePipeline(Krea2Pipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, Krea2Transformer transformer, QwenImageVaeDecoder vae, QwenImageVaeEncoder? vaeEncoder, bool isTurbo, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vae = vae;
        _vaeEncoder = vaeEncoder;
        _isTurbo = isTurbo;
        _loaders = loaders;
    }

    /// <summary>A Turbo/TDM checkpoint samples in 8 guidance-free steps, so it resolves against <see cref="Krea2Recipe.TurboDefaults"/> rather than Base's 28 steps at CFG 4.5.</summary>
    public ImageDefaults? VariantDefaults => _isTurbo ? Krea2Recipe.TurboDefaults : Krea2Recipe.FamilyDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        // Turbo: 8 steps, guidance off. Base: 28 steps, CFG 4.5.
        int steps = request.Steps ?? (_isTurbo ? Krea2Recipe.TurboDefaults.Steps : Krea2Recipe.FamilyDefaults.Steps);
        float cfg = _isTurbo ? Krea2Recipe.TurboDefaults.CfgScale : (request.CfgScale ?? Krea2Recipe.FamilyDefaults.CfgScale);
        bool useCfg = cfg > 1.0f;

        // Width/height must be multiples of 16 (2×2 patchify × 8× VAE), clamped to 128–4096.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = Math.Clamp(reqWidth / 16 * 16, 128, 4096);
        int height = Math.Clamp(reqHeight / 16 * 16, 128, 4096);
        if (width != reqWidth || height != reqHeight)
        {
            Logs.Info($"[Krea2RecipePipeline] Snapped {reqWidth}x{reqHeight} → {width}x{height} (multiple of 16, 128–4096).");
        }

        // TODO(E-IMG-4/5): LoRA, ControlNet, IP-Adapter are deferred.
        (int[] promptTokens, int promptDrop) = EncodeWithTemplate(_tokenizer, prompt);
        (int[] negTokens, int negDrop) = EncodeWithTemplate(_tokenizer, negative);

        // Resolved at the *snapped* size: Krea 2 rounds to a multiple of 16 above and the pipeline validates the
        // source against those same rounded dimensions.
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
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
        };

        RegionalPlan? regionalPlan = null;
        try
        {
            regionalPlan = BuildRegionalPlan(prompt, width, height, steps);

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
                promptTokens, useCfg ? negTokens : null, inner, bridge,
                promptDropIndex: promptDrop, negativeDropIndex: negDrop, regionalPlan: regionalPlan);

            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "krea2",
                    ["variant"] = _isTurbo ? "turbo" : "base",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
        }
    }

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c>
    /// parts, null otherwise (Tier 3.7). Each region's text is templated + drop-indexed the SAME way the base
    /// prompt is (<see cref="EncodeWithTemplate"/>) and encoded through <see cref="Krea2Pipeline.EncodeRegionText"/>
    /// — the same tapped-layer text encoder the base prompt uses. <see cref="RegionalPlan.BaseCond"/> is a required
    /// field on the resolver's signature that <see cref="Krea2Pipeline.GenerateFromTokens"/>'s regional path never
    /// reads (same as Flux.1/Flux.2 — confirmed by inspection) — a throwaway placeholder satisfies it.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            (int[] regionTokens, int regionDrop) = EncodeWithTemplate(_tokenizer, text);
            return _pipeline.EncodeRegionText(regionTokens, regionDrop);
        });
    }

    /// <summary>Builds the Krea 2 templated token sequence plus the prefix-drop index (the leading system-block + user-header positions whose hidden states the pipeline discards — Krea 2's <c>prompt_template_encode_start_idx</c>).</summary>
    private static (int[] tokens, int dropIndex) EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new List<int>(64);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw(Krea2SystemPrompt));
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
