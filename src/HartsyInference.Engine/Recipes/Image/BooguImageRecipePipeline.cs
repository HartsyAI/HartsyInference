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

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Boogu-Image pipeline driven against the native <see cref="ImageRequest"/>. Builds the Qwen3-VL chat-templated instruction tokens, encodes them through the 8B language tower under the loader's TE ⇄ DiT staging (evict the resident DiT, encode, free the encoder weights), and calls <see cref="BooguImagePipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>BooguImageLoader.Generate</c> text-to-image drive path.</summary>
public sealed unsafe class BooguImageRecipePipeline : IRecipePipeline
{
    /// <summary>Boogu T2I system prompt (verbatim from <c>pipeline_boogu.py</c> <c>SYSTEM_PROMPT_4_T2I_UNIFIED</c>).</summary>
    private const string SystemPromptT2I =
        "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

    private readonly BooguImagePipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly BooguImageTransformer _transformer;
    private readonly IBackend _backend;
    private readonly IReadOnlyList<SafeTensorsLoader> _loaders;

    // Prompt-embedding cache: repeat prompts skip the whole Qwen3-VL-8B encode (and the DiT eviction it forces —
    // the ~10 GB encoder and the ~10 GB fp8 DiT cannot coexist beside activations on 24 GB). Reusing the SAME
    // tensor references also keeps the transformer's ref-keyed refined-caption cache warm.
    private string? _cachedInstrKey;
    private Tensor? _cachedInstr;
    private string? _cachedNegKey;
    private Tensor? _cachedNeg;

    /// <summary>Wraps the constructed Boogu-Image pipeline plus its text stack, taking ownership of every disposable.</summary>
    public BooguImageRecipePipeline(BooguImagePipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, BooguImageTransformer transformer, IBackend backend, IReadOnlyList<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _transformer = transformer;
        _backend = backend;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? BooguImageRecipe.FamilyDefaults.Steps;
        // Text guidance from CFG scale (Boogu Base works well ~2–5; Turbo = 1). The negative prompt drives the uncond pass.
        float textGuidance = request.CfgScale ?? BooguImageRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4): reference-image editing (request.Img2Img) needs the Qwen3-VL vision tower + the pipeline's
        // EditFromEmbeddings path. Text-to-image only.

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
            Prompt = prompt,
            NegativePrompt = negative,
            Width = snappedW,
            Height = snappedH,
            Steps = steps,
            CfgScale = textGuidance,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
        };

        bool needNeg = textGuidance > 1.0f;
        bool instrHit = _cachedInstr is not null && _cachedInstrKey == prompt;
        bool negHit = !needNeg || (_cachedNeg is not null && _cachedNegKey == negative);
        if (!instrHit || !negHit)
        {
            _pipeline.EvictResidentWeights();
            if (!instrHit)
            {
                int[] tokens = BuildTemplatedTokens(_tokenizer, SystemPromptT2I, prompt);
                Tensor instrNew = _textEncoder.Encode(_backend, new[] { tokens });
                _ = instrNew.DataPointer;   // host-materialize: survives FreeActivations
                _cachedInstr?.Dispose();
                _cachedInstr = instrNew;
                _cachedInstrKey = prompt;
            }
            if (needNeg && (_cachedNeg is null || _cachedNegKey != negative))
            {
                int[] negTokens = BuildTemplatedTokens(_tokenizer, SystemPromptT2I, negative);
                Tensor negNew = _textEncoder.Encode(_backend, new[] { negTokens });
                _ = negNew.DataPointer;
                _cachedNeg?.Dispose();
                _cachedNeg = negNew;
                _cachedNegKey = negative;
            }
            _backend.Sync();
            _backend.FreeWeights(_textEncoder.EnumerateWeights());
            _backend.FreeActivations();
        }

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromEmbeddings(
            _cachedInstr!, inner, textGuidance, needNeg ? _cachedNeg : null, bridge);

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

    /// <summary>Builds the Qwen3-VL chat-templated token sequence (system preamble, user instruction, assistant header). Text fragments use the raw BPE; structural tokens are the Qwen special ids. The vision-start/image-pad span the edit path adds is omitted — this is the text-to-image form.</summary>
    private static int[] BuildTemplatedTokens(Qwen3Tokenizer tok, string system, string instruction)
    {
        List<int> ids = new List<int>(256);
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "system\n" + system);
        ids.Add(Qwen3Tokenizer.ImEndId);
        AppendRaw(ids, tok, "\n");
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "user\n");
        AppendRaw(ids, tok, instruction ?? "");
        ids.Add(Qwen3Tokenizer.ImEndId);
        AppendRaw(ids, tok, "\n");
        ids.Add(Qwen3Tokenizer.ImStartId);
        AppendRaw(ids, tok, "assistant\n");
        return ids.ToArray();
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
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
