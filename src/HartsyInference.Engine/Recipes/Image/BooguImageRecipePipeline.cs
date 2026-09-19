using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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

/// <summary>A constructed Boogu-Image pipeline driven against the native <see cref="ImageRequest"/>. Builds the Qwen3-VL chat-templated instruction tokens, encodes them through the 8B language tower under the loader's TE ⇄ DiT staging (evict the resident DiT, encode, free the encoder weights), and calls <see cref="BooguImagePipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>BooguImageLoader.Generate</c> text-to-image drive path. Wraps the constructed Boogu-Image pipeline plus its text stack, taking ownership of every disposable.</summary>
public sealed unsafe class BooguImageRecipePipeline(BooguImagePipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder,
    BooguImageTransformer transformer, IBackend backend, IReadOnlyList<IDisposable> componentSources,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    /// <summary>Boogu T2I system prompt (verbatim from <c>pipeline_boogu.py</c> <c>SYSTEM_PROMPT_4_T2I_UNIFIED</c>).</summary>
    private const string SystemPromptT2I =
        "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

    private readonly BooguImagePipeline _pipeline = pipeline;
    private readonly Qwen3Tokenizer _tokenizer = tokenizer;
    private readonly LlamaStyleEncoder _textEncoder = textEncoder;
    private readonly BooguImageTransformer _transformer = transformer;
    private readonly IBackend _backend = backend;
    private readonly IReadOnlyList<IDisposable> _componentSources = componentSources;
    private readonly MergedLoraStack? _loraStack = loraStack;

    // Prompt-embedding cache: repeat prompts skip the whole Qwen3-VL-8B encode (and the DiT eviction it forces —
    // the ~10 GB encoder and the ~10 GB fp8 DiT cannot coexist beside activations on 24 GB). Reusing the SAME
    // tensor references also keeps the transformer's ref-keyed refined-caption cache warm.
    private string? _cachedInstrKey;
    private Tensor? _cachedInstr;
    private string? _cachedNegKey;
    private Tensor? _cachedNeg;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? BooguImageRecipe.FamilyDefaults.Steps;
        // Text guidance from CFG scale (Boogu Base works well ~2–5; Turbo = 1). The negative prompt drives the uncond pass.
        float textGuidance = request.CfgScale ?? BooguImageRecipe.FamilyDefaults.CfgScale;

        // Reference editing runs at TEXT-ONLY guidance (imageGuidanceScale = 1), which BooguImagePipeline explicitly
        // supports by skipping the drop-all pass. Full image guidance additionally needs the Qwen3-VL vision tower to
        // produce a text-and-image-dropped embedding, which is still deferred — so the reference conditions the latent
        // stream, but its strength is not separately steerable yet.

        // Output size must be a multiple of 16 (2×2 patchify × 8× VAE).
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int snappedW = Math.Clamp(reqWidth / 16 * 16, 256, 2048);
        int snappedH = Math.Clamp(reqHeight / 16 * 16, 256, 2048);
        if (snappedW != reqWidth || snappedH != reqHeight)
        {
            Logs.Info($"[BooguImageRecipe] Snapped resolution {reqWidth}x{reqHeight} → {snappedW}x{snappedH} (multiple of 16, 256–2048).");
        }

        TextToImageRequest inner = new TextToImageRequest
        {
            SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
            Prompt = prompt,
            NegativePrompt = negative,
            Width = snappedW,
            Height = snappedH,
            Steps = steps,
            CfgScale = textGuidance,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
            // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
        };

        // The edit path needs a drop-text embedding unconditionally (EditFromEmbeddings takes it non-null), so an
        // edit request forces the negative encode even at guidance 1.
        (int snapW2, int snapH2) = (snappedW, snappedH);
        using Img2ImgResolver.Img2ImgSpec? refEdit = RecipeImg2ImgBinder.Resolve(request, snapW2, snapH2);
        bool needNeg = textGuidance > 1.0f || refEdit is not null;
        // Declaring CondScale is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar
        // now. The cache below is keyed on the prompt STRING, so a weighted and an unweighted prompt are already
        // different keys — but a repeat of the SAME weighted prompt would hit, so the scale still has to land on
        // a per-request copy or it would compound generation after generation.
        WeightedTokenSequence instrSequence = EncodeWeighted(prompt);
        WeightedTokenSequence negSequence = EncodeWeighted(negative);
        bool instrHit = _cachedInstr is not null && _cachedInstrKey == prompt;
        bool negHit = !needNeg || (_cachedNeg is not null && _cachedNegKey == negative);
        if (!instrHit || !negHit)
        {
            _pipeline.EvictResidentWeights();
            if (!instrHit)
            {
                Tensor instrNew = _textEncoder.Encode(_backend, new[] { instrSequence.Tokens });
                _ = instrNew.DataPointer;   // host-materialize: survives FreeActivations
                _cachedInstr?.Dispose();
                _cachedInstr = instrNew;
                _cachedInstrKey = prompt;
            }
            if (needNeg && (_cachedNeg is null || _cachedNegKey != negative))
            {
                Tensor negNew = _textEncoder.Encode(_backend, new[] { negSequence.Tokens });
                _ = negNew.DataPointer;
                _cachedNeg?.Dispose();
                _cachedNeg = negNew;
                _cachedNegKey = negative;
            }
            _backend.Sync();
            _backend.FreeWeights(_textEncoder.EnumerateWeights());
            _backend.FreeActivations();
        }

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        Tensor? weightedInstr = CondTokenWeights.Apply(_backend, _cachedInstr!, null, instrSequence).Cond;
        Tensor instrEmbeddings = weightedInstr ?? _cachedInstr!;
        Tensor? weightedNeg = needNeg && _cachedNeg is not null
            ? CondTokenWeights.Apply(_backend, _cachedNeg, null, negSequence).Cond : null;
        Tensor? negEmbeddings = weightedNeg ?? (needNeg ? _cachedNeg : null);
        try
        {

        // IP2P image guidance: drop-text and drop-all share the same "no text" embedding with the text-only
        // encoder; the image drop happens transformer-side via refLatents: null on the drop-all forward.
        float imageGuidance = (float)(request.InstructPix2PixCfg ?? 1.0);
        (byte[] rgb, int outW, int outH, int usedSeed) = refEdit is null
            ? _pipeline.GenerateFromEmbeddings(
                instrEmbeddings, inner, textGuidance, negEmbeddings, bridge)
            : _pipeline.EditFromEmbeddings(
                instrEmbeddings, negEmbeddings!, dropAllEmbeddings: imageGuidance > 1f ? negEmbeddings : null,
                [refEdit.SourceTensor], inner,
                textGuidanceScale: textGuidance, imageGuidanceScale: imageGuidance, onProgress: bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "boogu",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = textGuidance.ToString(CultureInfo.InvariantCulture),
            },
        };
        }
        finally
        {
            // Per-request copies, not the cached originals.
            weightedInstr?.Dispose();
            weightedNeg?.Dispose();
        }
    }

    /// <summary>The Boogu chat-templated sequence plus its per-token weights. Boogu already tokenizes the
    /// instruction separately from the template text, so the weighted build reproduces the unweighted ids
    /// exactly; routing through <see cref="TemplatedPromptTokens"/> keeps that a guarantee rather than a
    /// coincidence of how <see cref="BuildTemplatedTokens"/> happens to be written.</summary>
    private WeightedTokenSequence EncodeWeighted(string prompt)
    {
        (int[] prefix, int[] suffix) = TemplateIds(_tokenizer, SystemPromptT2I);
        return TemplatedPromptTokens.Build(PromptTagFlattening.Flatten(prompt),
            t => BuildTemplatedTokens(_tokenizer, SystemPromptT2I, t), _tokenizer.EncodeRaw, prefix, suffix);
    }

    /// <summary>The ids <see cref="BuildTemplatedTokens"/> puts either side of the instruction.</summary>
    private static (int[] Prefix, int[] Suffix) TemplateIds(Qwen3Tokenizer tok, string system)
    {
        List<int> prefix = new List<int>(256) { Qwen3Tokenizer.ImStartId };
        AppendRaw(prefix, tok, "system\n" + system);
        prefix.Add(Qwen3Tokenizer.ImEndId);
        AppendRaw(prefix, tok, "\n");
        prefix.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(prefix, tok, "user\n");
        List<int> suffix = new List<int>(8) { Qwen3Tokenizer.ImEndId };
        AppendRaw(suffix, tok, "\n");
        suffix.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(suffix, tok, "assistant\n");
        return (prefix.ToArray(), suffix.ToArray());
    }

    /// <summary>Builds the Qwen3-VL chat-templated token sequence (system preamble, user instruction, assistant header). Text fragments use the raw BPE; structural tokens are the Qwen special ids. The vision-start/image-pad span the edit path adds is omitted — this is the text-to-image form.</summary>
    private static int[] BuildTemplatedTokens(Qwen3Tokenizer tok, string system, string instruction)
    {
        // Assembled from TemplateIds rather than repeating it, so the weighted path — which splices the prompt
        // between those same two halves — cannot drift from this one. The instruction was always its own
        // EncodeRaw call, which is why splicing reproduces these ids exactly.
        (int[] prefix, int[] suffix) = TemplateIds(tok, system);
        return [.. prefix, .. tok.EncodeRaw(instruction ?? ""), .. suffix];
    }

    /// <summary>Appends the byte-level BPE ids for <paramref name="text"/> (no special-token handling).</summary>
    private static void AppendRaw(List<int> dst, Qwen3Tokenizer tok, string text)
    {
        foreach (int id in tok.EncodeRaw(text))
        {
            dst.Add(id);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _cachedInstr?.Dispose();
        _cachedNeg?.Dispose();
        _pipeline.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        _tokenizer.Dispose();
        foreach (IDisposable source in _componentSources)
        {
            source.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
