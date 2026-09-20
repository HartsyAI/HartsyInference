using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Lumina-2 pipeline driven against the native <see cref="ImageRequest"/>. Unlike the other recipes, the Gemma-2 text-encoder forward is NOT owned by the pipeline (HartsyInference's Gemma-2 encoder is an approximation), so this holds the encoder + tokenizer and produces the <c>hidden_states[-2]</c> caption embeddings itself (system-prompt-prefixed, one plain string, real tokens only) — with a one-slot cond/negative cache — then calls <see cref="Lumina2Pipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>Lumina2Loader.Generate</c> text-to-image drive path. Wraps the constructed Lumina-2 pipeline plus its text encoder + tokenizer, taking ownership of every disposable. The backend is retained because the caption encode runs outside the pipeline.</summary>
public sealed class Lumina2RecipePipeline(Lumina2Pipeline pipeline, IBackend backend, LlamaStyleEncoder textEncoder,
    GemmaTokenizer tokenizer, Lumina2Transformer transformer, string systemPrompt, IDisposable checkpoint,
    IReadOnlyList<IDisposable> sideModelLoaders,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly Lumina2Pipeline _pipeline = pipeline;
    private readonly IBackend _backend = backend;
    private readonly LlamaStyleEncoder _textEncoder = textEncoder;
    private readonly GemmaTokenizer _tokenizer = tokenizer;
    private readonly Lumina2Transformer _transformer = transformer;
    private readonly string _systemPrompt = systemPrompt;
    private readonly IDisposable _checkpoint = checkpoint;
    private readonly IReadOnlyList<IDisposable> _sideModelLoaders = sideModelLoaders;

    // Single-slot prompt-embedding caches (cond + negative), keyed on the raw prompt text — repeat prompts skip
    // the Gemma-2 forward. Host-materialized so activation reclaims can't revert them.
    private string? _cachedPrompt;
    private Tensor? _cachedEmbeds;
    private string? _cachedNegPrompt;
    private Tensor? _cachedNegEmbeds;

    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        // <=0 defers to the Lumina-2 reference default (4.0, always runs a negative pass); a positive value overrides.
        float cfg = request.CfgScale ?? Lumina2Recipe.FamilyDefaults.CfgScale;
        int steps = request.Steps ?? Lumina2Recipe.FamilyDefaults.Steps;

        Tensor condEmbeds = GetOrEncode(prompt);
        // The pipeline requires negative embeddings whenever cfg > 1 (the reference always runs a negative pass at
        // its default cfg=4) — the empty prompt through the same template is the reference uncond.
        Tensor? negEmbeds = cfg > 1.0f ? GetOrEncodeNegative(negative) : null;

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromEmbeddings(
            condEmbeds, inner,
            cfgScale: cfg,
            negativeCaptionEmbeddings: negEmbeds,
            onProgress: bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "lumina2",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Returns the [1, T, 2304] <c>hidden_states[-2]</c> conditioning for the prompt (one-slot cache).</summary>
    private Tensor GetOrEncode(string prompt)
    {
        if (_cachedEmbeds is not null && _cachedPrompt == prompt)
        {
            return _cachedEmbeds;
        }
        Tensor embeds = EncodeTemplated(prompt);
        _cachedEmbeds?.Dispose();
        _cachedEmbeds = embeds;
        _cachedPrompt = prompt;
        return embeds;
    }

    /// <summary>Negative-prompt twin of <see cref="GetOrEncode"/>.</summary>
    private Tensor GetOrEncodeNegative(string negative)
    {
        if (_cachedNegEmbeds is not null && _cachedNegPrompt == negative)
        {
            return _cachedNegEmbeds;
        }
        Tensor embeds = EncodeTemplated(negative);
        _cachedNegEmbeds?.Dispose();
        _cachedNegEmbeds = embeds;
        _cachedNegPrompt = negative;
        return embeds;
    }

    /// <summary>Tokenizes the system-prompt-prefixed caption and taps <c>hidden_states[-2]</c> (encoder layer index NumLayers-1, WITHOUT the final RMSNorm), host-materializing the result so it survives activation reclaims. Applies the prompt's per-token weights via ComfyBlend.</summary>
    /// <remarks>Lumina-2 does NOT pad — <c>Gemma2BTokenizer</c> sets <c>pad_to_max_length=False</c> with
    /// <c>min_length=1</c> — so the conditioning length tracks the prompt and the empty baseline has to be rebuilt
    /// at that length for every prompt instead of encoded once and reused.
    /// <para>Caching the BLENDED result is safe because both caches are keyed on the raw prompt STRING, emphasis
    /// grammar included: <c>(fox:1.5)</c> and <c>fox</c> are already different keys. A token-id key would collide,
    /// since the emphasis is stripped before tokenization and does not change the ids.</para></remarks>
    private unsafe Tensor EncodeTemplated(string prompt)
    {
        WeightedTokenSequence sequence = TemplatedPromptTokens.Build(
            PromptTagFlattening.Flatten(prompt),
            t => _tokenizer.Encode(_systemPrompt + " <Prompt Start> " + t),
            EncodeSpan, TemplatePrefix, []).Truncate(_tokenizer.MaxLength);
        int[] tokens = sequence.Tokens;
        // Round-trip decode is the cheapest way to catch a wrong-vocab tokenizer: mismatched ids stay in range
        // and produce coherent-but-unrelated conditioning instead of throwing.
        Logs.Debug($"[Lumina2] caption tokens={tokens.Length} ids[0..8]=[{string.Join(",", tokens[..Math.Min(8, tokens.Length)])}] " +
            $"roundtrip=\"{_tokenizer.Decode(tokens)}\"");
        int tapIndex = _textEncoder.NumLayers - 1;
        Tensor embeds = _textEncoder.EncodeMultiLayer(_backend, new[] { tokens }, new[] { tapIndex });
        if (!sequence.IsUniformlyUnweighted)
        {
            // start + pad, read off ComfyUI rather than assumed: `gen_empty_tokens` emits start + end + padding and
            // `Gemma2_2BModel` declares `special_tokens={"start": 2, "pad": 0}` with no end, so the baseline is BOS
            // followed by pad. Row 0 is a real prompt position here, so getting that first id wrong would shift the
            // blend for any prompt whose first word is weighted.
            int[] empty = new int[tokens.Length];
            empty[0] = GemmaTokenizer.BosTokenId;
            using Tensor emptyEmbeds = _textEncoder.EncodeMultiLayer(_backend, new[] { empty }, new[] { tapIndex });
            if (ComfyBlend.Apply(_backend, embeds, emptyEmbeds, sequence.Weights) is Tensor blended)
            {
                embeds.Dispose();
                embeds = blended;
            }
        }
        _ = embeds.DataPointer;
        return embeds;
    }

    /// <summary>One weighted span's ids, BOS stripped. ComfyUI tokenizes each word alone and slices from
    /// <c>tokens_start=1</c> (<c>sd1_clip.py:501</c>, <c>has_start_token</c> defaulting true for
    /// <c>Gemma2BTokenizer</c>), then prepends the start token once for the whole batch — so a per-span BOS would
    /// put a stray sentence start in the middle of the caption.</summary>
    private IReadOnlyList<int> EncodeSpan(string text)
    {
        IReadOnlyList<int> ids = _tokenizer.EncodeRaw(text);
        return ids.Count > 0 && ids[0] == GemmaTokenizer.BosTokenId ? [.. ids.Skip(1)] : ids;
    }

    /// <summary>The ids the templated encode puts before the caption: the single batch-level BOS plus the system
    /// prompt and its <c>&lt;Prompt Start&gt;</c> marker, all pinned to weight 1.</summary>
    private int[] TemplatePrefix =>
        field ??= [GemmaTokenizer.BosTokenId, .. EncodeSpan(_systemPrompt + " <Prompt Start> ")];

    /// <inheritdoc/>
    public void Dispose()
    {
        _cachedEmbeds?.Dispose();
        _cachedNegEmbeds?.Dispose();
        _pipeline.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        _checkpoint.Dispose();
        foreach (IDisposable loader in _sideModelLoaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
