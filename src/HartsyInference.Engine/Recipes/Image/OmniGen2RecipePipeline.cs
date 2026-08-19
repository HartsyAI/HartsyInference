using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
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

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed OmniGen 2 pipeline driven against the native <see cref="ImageRequest"/>. The Qwen2.5-VL-3B forward lives outside <see cref="OmniGen2Pipeline"/>, so this owns the encoder: it live-encodes the ComfyUI chat template (full sequence, no prefix drop, final hidden state after the last RMSNorm), frees the encoder's device weights, then runs <see cref="OmniGen2Pipeline.GenerateFromEmbeddings"/>. Two single-slot embedding caches (positive + negative) let seed-only reruns skip the encoder entirely, as the SwarmUI backend's <c>OmniGen2CacheEntry.GetOrEncode</c> did.</summary>
public sealed class OmniGen2RecipePipeline : IRecipePipeline
{
    /// <summary>ComfyUI's OmniGen2 system prompt, verbatim (<c>comfy/text_encoders/omnigen2.py</c> llama_template).</summary>
    private const string SystemPrompt =
        "You are a helpful assistant that generates high-quality images based on user instructions.";

    /// <summary>ComfyUI truncates the templated sequence at 512 tokens.</summary>
    private const int MaxTokens = 512;

    private readonly OmniGen2Pipeline _pipeline;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly LlamaStyleEncoder _textEncoder;
    private readonly OmniGen2Transformer _transformer;
    private readonly IBackend _backend;
    private readonly List<SafeTensorsLoader> _loaders;

    private string? _cachedPrompt;
    private Tensor? _cachedEmbeds;
    private string? _cachedNegPrompt;
    private Tensor? _cachedNegEmbeds;

    /// <summary>Wraps the constructed OmniGen 2 pipeline plus its text stack, taking ownership of every disposable.</summary>
    private readonly MergedLoraStack? _loraStack;

    public OmniGen2RecipePipeline(OmniGen2Pipeline pipeline, Qwen3Tokenizer tokenizer, LlamaStyleEncoder textEncoder, OmniGen2Transformer transformer, IBackend backend, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
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
        int steps = request.Steps ?? OmniGen2Recipe.FamilyDefaults.Steps;
        // OmniGen 2's own default is guidance-free; the SwarmUI loader mapped a non-positive CFG to 1.0.
        float cfg = request.CfgScale ?? OmniGen2Recipe.FamilyDefaults.CfgScale;
        bool needNegative = cfg > 1.0f;

        // TODO(E-IMG-4/5): Prompt Images as additional references, align_res output sizing, LoRA, ControlNet and
        // regional prompting are deferred, as is the pipeline's cfgRange gate (left at its full-schedule default).
        EnsureEmbeddings(prompt, needNegative ? negative : null);

        // Reference editing, not img2img: the init image becomes in-context reference latents that condition every
        // step, so there is no denoise-strength start step and Creativity does not apply.
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? refEdit = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);

        TextToImageRequest inner = new TextToImageRequest
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
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
        };

        (byte[] rgb, int outW, int outH, int usedSeed) = refEdit is null
            ? _pipeline.GenerateFromEmbeddings(
                _cachedEmbeds!,
                inner,
                cfgScale: cfg,
                negativeCaptionEmbeddings: needNegative ? _cachedNegEmbeds : null,
                textGuidanceScale: cfg,
                onProgress: bridge)
            : _pipeline.EditFromEmbeddings(
                _cachedEmbeds!,
                needNegative ? _cachedNegEmbeds : null,
                [refEdit.SourceTensor],
                inner,
                textGuidanceScale: cfg,
                imageGuidanceScale: (float)(request.InstructPix2PixCfg ?? 2.0),
                onProgress: bridge);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "omnigen2",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Refreshes the cached caption embeddings for this request, running the 3B encoder only on a cache miss and freeing its device weights afterwards (the DiT needs that headroom). <paramref name="negative"/> is null when guidance is off.</summary>
    private void EnsureEmbeddings(string prompt, string? negative)
    {
        bool promptHit = _cachedEmbeds is not null && string.Equals(_cachedPrompt, prompt, StringComparison.Ordinal);
        bool negativeHit = negative is null || (_cachedNegEmbeds is not null && string.Equals(_cachedNegPrompt, negative, StringComparison.Ordinal));
        if (promptHit && negativeHit)
        {
            return;
        }

        _backend.PreloadWeights(_textEncoder.EnumerateWeights());
        if (!promptHit)
        {
            Tensor embeds = _textEncoder.Encode(_backend, new[] { EncodeWithTemplate(_tokenizer, prompt) });
            _cachedEmbeds?.Dispose();
            _cachedEmbeds = embeds;
            _cachedPrompt = prompt;
        }
        if (negative is not null && !negativeHit)
        {
            Tensor negEmbeds = _textEncoder.Encode(_backend, new[] { EncodeWithTemplate(_tokenizer, negative) });
            _cachedNegEmbeds?.Dispose();
            _cachedNegEmbeds = negEmbeds;
            _cachedNegPrompt = negative;
        }
        _backend.Sync();
        // Host-materialize before the activation sweep so the cached embeddings survive across generations.
        MaterializeOnHost(_cachedEmbeds);
        MaterializeOnHost(_cachedNegEmbeds);
        _backend.FreeWeights(_textEncoder.EnumerateWeights());
        _backend.FreeActivations();
    }

    /// <summary>Touches the tensor's host pointer so its contents are pulled back from the device — a device-side reclaim can then no longer invalidate the cached embedding.</summary>
    private static unsafe void MaterializeOnHost(Tensor? tensor)
    {
        if (tensor is not null)
        {
            _ = tensor.DataPointer;
        }
    }

    /// <summary>Tokenizes with ComfyUI's OmniGen2 chat template. The FULL sequence (system block included) is the conditioning — no prefix drop, unlike Qwen-Image. Special tokens are inserted by id; text segments are BPE'd separately, which matches HF tokenization because special tokens split the text at exactly these boundaries.</summary>
    private static int[] EncodeWithTemplate(Qwen3Tokenizer tokenizer, string prompt)
    {
        List<int> ids = new List<int>(96);
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("system\n" + SystemPrompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("user\n" + prompt));
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        if (ids.Count > MaxTokens)
        {
            ids.RemoveRange(MaxTokens, ids.Count - MaxTokens);
        }
        return ids.ToArray();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _cachedEmbeds?.Dispose();
        _cachedNegEmbeds?.Dispose();
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _textEncoder.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
