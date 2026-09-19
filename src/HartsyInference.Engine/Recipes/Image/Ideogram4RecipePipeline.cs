using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Ideogram 4 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Ideogram4Pipeline"/> owns the Qwen3-VL forward, so this only chat-templates and trims the prompt tokens, snaps the resolution to Ideogram's 16-pixel grid, and maps <see cref="ImageRequest.Steps"/> onto the nearest official sampler preset (the preset carries the per-step asymmetric-CFG guidance schedule, so CfgScale and the negative prompt are ignored by design). Mirrors the SwarmUI backend's <c>Ideogram4Loader.Generate</c> drive path. Wraps the constructed Ideogram 4 pipeline plus its tokenizer and both transformers, taking ownership of every disposable.</summary>
public sealed class Ideogram4RecipePipeline(Ideogram4Pipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder,
    Ideogram4Transformer conditional, Ideogram4Transformer unconditional, IReadOnlyList<IDisposable> componentSources,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly Ideogram4Pipeline _pipeline = pipeline;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly LlamaStyleEncoder _textEncoder = textEncoder;
    private readonly Ideogram4Transformer _conditional = conditional;
    private readonly Ideogram4Transformer _unconditional = unconditional;
    private readonly IReadOnlyList<IDisposable> _componentSources = componentSources;
    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;

        // Width/height must be multiples of 16 (2x2 patchify x 8x VAE), range 256-2048.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int snappedW = Math.Clamp(reqWidth / 16 * 16, 256, 2048);
        int snappedH = Math.Clamp(reqHeight / 16 * 16, 256, 2048);
        if (snappedW != reqWidth || snappedH != reqHeight)
        {
            Logs.Info($"[Ideogram4] Snapped resolution {reqWidth}x{reqHeight} -> {snappedW}x{snappedH}.");
        }

        // Steps pick the nearest official preset; the preset owns the guidance schedule (gw~7 main + gw~3 polish)
        // and the logit-normal mu/std, so CfgScale and the negative prompt are intentionally not consumed.
        int steps = request.Steps ?? Ideogram4Recipe.FamilyDefaults.Steps;
        Ideogram4SamplerPreset preset = steps <= 14 ? Ideogram4SamplerPreset.Turbo12
            : steps >= 40 ? Ideogram4SamplerPreset.Quality48
            : Ideogram4SamplerPreset.Default20;
        if (steps != preset.NumSteps)
        {
            Logs.Info($"[Ideogram4] Steps={steps} mapped to official preset {preset.Name} ({preset.NumSteps} steps — Ideogram 4 uses fixed preset schedules).");
        }
        if (!string.IsNullOrWhiteSpace(request.NegativePrompt))
        {
            Logs.Info("[Ideogram4] Negative prompt is ignored — Ideogram 4's asymmetric CFG runs the unconditional branch with zeroed text features.");
        }

        // TODO(E-IMG-5): the SwarmUI host wrapped a plain prompt into Ideogram's structured JSON caption shape
        // (optionally via an LLM "magic prompt" expansion). That is a host-side prompt helper and is NOT ported —
        // the prompt is fed verbatim. Ideogram 4 was trained only on structured JSON captions, so a bare prompt
        // frequently trips its built-in safety filter (grey placeholder output); supply a structured caption.
        Logs.Warning("[Ideogram4] Prompt fed verbatim — supply a structured JSON caption for reliable results (magic-prompt expansion is host-side and not wired here).");

        // includeThinkBlock:false — Ideogram's encoder is Qwen3-VL-8B-Instruct, whose chat template ends the
        // generation prompt at "<|im_start|>assistant\n" with no <think> block. EncodeChat right-pads to maxLength
        // with BOS; feeding ~2048 mostly-pad tokens would dilute conditioning and multiply attention cost.
        // Declaring CondScale is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar.
        // The base encode covers the region tags too, so a region's own weight would surface as a base weight —
        // RegionalPromptWeightSplit is where that question lives.
        bool hasRegionParts = RegionalPromptResolver.HasRegionParts(prompt);
        if (hasRegionParts && RegionalPromptWeightSplit.BaseTextCarriesWeight(prompt))
        {
            throw new NotSupportedException(
                "Ideogram 4 cannot weight the base prompt and a region in the same request: the base encode "
                + "covers the region tags too, so the two sets of weights would land on the same conditioning "
                + "rows. Move the emphasis inside the region, or drop the region tags.");
        }
        WeightedTokenSequence promptSequence =
            EncodeWeighted(RegionalPromptWeightSplit.BaseText(prompt, hasRegionParts));
        int[] promptTokens = promptSequence.Tokens;

        // Regional/object prompt parts, chat-templated + encoded via the SAME Qwen3-VL multi-layer tap
        // configuration as the base prompt above (Ideogram4Pipeline.EncodeRegionText).
        using Tensor? baseCondPlaceholder = hasRegionParts ? new Tensor(new TensorShape(1), DType.F32) : null;
        RegionalPlan? regionalPlan = baseCondPlaceholder is null ? null
            : RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, snappedW, snappedH, preset.NumSteps, encodeRegion: text =>
            {
                // Each region is its own leaf carrying its own emphasis, as encode_leaves does per region.
                WeightedTokenSequence region = EncodeWeighted(PromptTagFlattening.Flatten(text));
                return _pipeline.EncodeRegionText(region.Tokens, region);
            });

        // TODO(E-IMG-4): img2img/inpaint, LoRA, ControlNet, IP-Adapter, refiner and ImageRequest.Components
        // overrides are deferred — text-to-image (+ regional prompting, above) only.
        // Resolved at the snapped size the pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, snappedW, snappedH);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = "",
                Width = snappedW,
                Height = snappedH,
                Steps = preset.NumSteps,
                CfgScale = 7.0f,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        byte[] rgb; int outW, outH, usedSeed;
        try
        {
            (rgb, outW, outH, usedSeed) = _pipeline.GenerateFromTokens(
                promptTokens, inner, preset, bridge, regionalPlan: regionalPlan, promptWeights: promptSequence);
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
        }

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "ideogram4",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = preset.NumSteps.ToString(CultureInfo.InvariantCulture),
                ["preset"] = preset.Name,
            },
        };
    }

    /// <summary>Strips the trailing run of <paramref name="padId"/> tokens <see cref="Qwen3Tokenizer.EncodeChat"/> right-pads with, keeping everything up to and including the last real token.</summary>
    private static int[] TrimRightPad(int[] tokens, int padId)
    {
        int end = tokens.Length;
        while (end > 1 && tokens[end - 1] == padId)
        {
            end--;
        }
        return end == tokens.Length ? tokens : tokens[..end];
    }

    /// <summary>The Ideogram 4 chat-templated sequence plus its per-token weights. An unweighted prompt keeps
    /// <see cref="Qwen3Tokenizer.EncodeChat"/> (right-pad trimmed, as before) so its ids are exactly what they
    /// were before weighting existed — the template merges <c>user\n</c> with the prompt in one BPE call, which a
    /// per-span build cannot reproduce for a prompt that starts with whitespace.</summary>
    private WeightedTokenSequence EncodeWeighted(string prompt)
    {
        (int[] prefix, int[] suffix) = _tokenizer.ChatTemplateIds(includeThinkBlock: false);
        return TemplatedPromptTokens.Build(prompt, Templated, _tokenizer.EncodeRaw, prefix, suffix)
            .Truncate(_tokenizer.MaxLength);

        int[] Templated(string text) =>
            TrimRightPad(_tokenizer.EncodeChat(text, includeThinkBlock: false), Qwen3Tokenizer.BosTokenId);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _conditional.Dispose();
        _unconditional.Dispose();
        foreach (IDisposable source in _componentSources)
        {
            source.Dispose();
        }
        // Last: the stack owns the merged weight tensors both transformers were serving.
        _loraStack?.Dispose();
    }
}
