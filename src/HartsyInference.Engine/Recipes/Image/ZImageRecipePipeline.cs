using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Z-Image pipeline driven against the native <see cref="ImageRequest"/>. Owns the Qwen3-4B encoder + tokenizer (the text-encoder forward lives outside <see cref="ZImagePipeline"/>): it encodes the prompt (and, for CFG, the negative) into caption embeddings, then runs <see cref="ZImagePipeline.GenerateFromEmbeddings"/>. Mirrors the SwarmUI backend's <c>ZImageLoader.Generate</c> drive path. Z-Image's non-standard CFG combine and mandatory velocity negation are encoded inside the pipeline; this just calls it with the right cfg + embeddings.</summary>
public sealed unsafe class ZImageRecipePipeline : IRecipePipeline
{
    private readonly ZImagePipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly ZImageTransformer _transformer;
    private readonly VaeDecoder _vae;
    private readonly VaeEncoder _vaeEncoder;
    private readonly Tensor[] _transformerWeightTensors;
    private readonly Tensor[] _qwenWeightTensors;
    private readonly Tensor[] _ownedVaeWeights;
    private readonly IBackend _backend;
    private readonly IBackend _textBackend;
    private readonly ImageDefaults _variantDefaults;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly SafeTensorsLoader _qwenLoader;
    private readonly SafeTensorsLoader _vaeLoader;
    private int _disposed;

    // Last-used prompt embeddings. They are host-materialized before the Qwen phase is reclaimed and retain the
    // same object identity across cache hits, which also keeps ZImageTransformer's reference-keyed refined-caption
    // cache warm. The exact unpadded token sequence is the cache key, including any legitimate end-of-text ID.
    private int[]? _cachedPositiveKey;
    private Tensor? _cachedPositive;
    private int[]? _cachedNegativeKey;
    private Tensor? _cachedNegative;

    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed Z-Image pipeline plus its text stack, taking ownership of every disposable.</summary>
    public ZImageRecipePipeline(ZImagePipeline pipeline, LlamaStyleEncoder qwen, Qwen3Tokenizer tokenizer,
        ZImageTransformer transformer, VaeDecoder vae, VaeEncoder vaeEncoder, Tensor[] transformerWeightTensors,
        Tensor[] qwenWeightTensors, Tensor[] ownedVaeWeights, IBackend backend, IBackend textBackend,
        ImageDefaults variantDefaults, SafeTensorsLoader checkpointLoader, SafeTensorsLoader qwenLoader,
        SafeTensorsLoader vaeLoader, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _pipeline = pipeline;
        _qwen = qwen;
        _tokenizer = tokenizer;
        _transformer = transformer;
        _vae = vae;
        _vaeEncoder = vaeEncoder;
        _transformerWeightTensors = transformerWeightTensors;
        _qwenWeightTensors = qwenWeightTensors;
        _ownedVaeWeights = ownedVaeWeights;
        _backend = backend;
        _textBackend = textBackend;
        _variantDefaults = variantDefaults;
        _checkpointLoader = checkpointLoader;
        _qwenLoader = qwenLoader;
        _vaeLoader = vaeLoader;
    }

    /// <inheritdoc/>
    public ImageDefaults? VariantDefaults => _variantDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _variantDefaults.Steps;
        // Z-Image CFG: <=0 (or 1) means Turbo (no CFG, single forward). >1 is Base and requires a negative encode.
        float cfg = request.CfgScale ?? _variantDefaults.CfgScale;
        bool needNegative = cfg > 1.0f;
        int penultimateIdx = _qwen.NumLayers - 1;

        // Validate/map every request-level input before a cache miss can evict the resident DiT and run Qwen.
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
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        // Upstream Z-Image requests enable_thinking=true. Qwen's template then stops at the assistant prefix;
        // includeThinkBlock is deliberately inverse-named because the empty block is emitted only when thinking
        // is disabled.
        (int[] paddedTokenIds, int tokenCount) =
            _tokenizer.EncodeChatWithLength(prompt, includeThinkBlock: false);
        int[] tokenIds = paddedTokenIds[..tokenCount];
        int[]? negTokens = null;
        if (needNegative)
        {
            // Encode even an empty negative — the reference passes "" through the encoder, yielding the short but
            // valid unconditional embedding CFG needs.
            (int[] paddedNegativeIds, int negativeCount) =
                _tokenizer.EncodeChatWithLength(negative, includeThinkBlock: false);
            negTokens = paddedNegativeIds[..negativeCount];
        }

        RegionalPlan? regionalPlan = null;
        try
        {
            regionalPlan = EnsurePromptCache(
                tokenIds, negTokens, needNegative, penultimateIdx, prompt, reqWidth, reqHeight, steps);
            Tensor positiveEmbeddings = _cachedPositive!;
            Tensor? negativeEmbeddings = needNegative ? _cachedNegative : null;

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
            };

            (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromEmbeddings(
                positiveEmbeddings,
                inner,
                cfgScale: cfg,
                negativeCaptionEmbeddings: negativeEmbeddings,
                onProgress: bridge,
                regionalPlan: regionalPlan);

            return new ImageResult
            {
                Rgb = rgb,
                Width = width,
                Height = height,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "zimage",
                    ["size"] = $"{width}x{height}",
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

    private RegionalPlan? EnsurePromptCache(
        int[] positiveTokens,
        int[]? negativeTokens,
        bool needNegative,
        int layerIndex,
        string rawPrompt,
        int width,
        int height,
        int steps)
    {
        bool needRegions = RegionalPromptResolver.HasRegionParts(rawPrompt);
        bool positiveHit = _cachedPositive is not null && _cachedPositiveKey is not null
            && _cachedPositiveKey.AsSpan().SequenceEqual(positiveTokens);
        bool negativeHit = !needNegative || (_cachedNegative is not null && _cachedNegativeKey is not null
            && _cachedNegativeKey.AsSpan().SequenceEqual(negativeTokens!));
        if (positiveHit && negativeHit && !needRegions)
        {
            Logs.Info("[Z-Image] prompt-embedding cache hit — Qwen phase skipped");
            return null;
        }

        // On a dedicated TE device, Qwen does not contend with the primary DiT. On the common same-device path,
        // evict only Z-Image's promoted weights and SDPA workspaces before asking the card for the 8 GB encoder.
        if (ReferenceEquals(_textBackend, _backend))
            _pipeline.EvictResidentWeights();
        // On the primary card the DiT eviction above leaves enough room for the 8.0 GB Qwen checkpoint plus the
        // 335 MB decoder (verified on the 12 GB production floor), so retain that warm decoder. A shared secondary
        // TE/VAE card has no corresponding DiT eviction and must reclaim the prior VAE phase before Qwen preload.
        if (!ReferenceEquals(_textBackend, _backend) && ReferenceEquals(_textBackend, _pipeline.VaeBackend))
            _pipeline.EvictVaeDeviceState();

        Tensor? newPositive = null;
        Tensor? newNegative = null;
        int[]? newPositiveKey = null;
        int[]? newNegativeKey = null;
        RegionalPlan? regionalPlan = null;
        List<Tensor>? regionalCandidates = needRegions ? [] : null;
        Exception? phaseError = null;
        try
        {
            try
            {
                _textBackend.PreloadWeights(_qwen.EnumerateWeights());
                if (!positiveHit)
                {
                    newPositive = EncodePrompt(positiveTokens, layerIndex);
                    _ = newPositive.DataPointer;
                    newPositiveKey = (int[])positiveTokens.Clone();
                }
                if (needNegative && !negativeHit)
                {
                    newNegative = EncodePrompt(negativeTokens!, layerIndex);
                    _ = newNegative.DataPointer;
                    newNegativeKey = (int[])negativeTokens!.Clone();
                }
                if (needRegions)
                {
                    // Region captions share the exact tokenizer/template, tapped Qwen layer, placement backend,
                    // and weight-residency phase used by the cached global captions.
                    Tensor baseConditioning = newPositive ?? _cachedPositive!;
                    regionalPlan = RegionalPromptResolver.Resolve(
                        rawPrompt, baseConditioning, width, height, steps, encodeRegion: text =>
                        {
                            (int[] paddedRegionIds, int regionCount) =
                                _tokenizer.EncodeChatWithLength(text, includeThinkBlock: false);
                            Tensor region = EncodePrompt(paddedRegionIds[..regionCount], layerIndex);
                            regionalCandidates!.Add(region);
                            _ = region.DataPointer;
                            return region;
                        });
                }
            }
            catch (Exception ex)
            {
                phaseError = ex;
                throw;
            }
            finally
            {
                Exception? cleanupError = CleanupTextEncoderPhase();
                if (cleanupError is not null)
                {
                    if (phaseError is null)
                    {
                        throw new InvalidOperationException(
                            "Z-Image text encoding completed, but its device phase could not be reclaimed.", cleanupError);
                    }
                    Logs.Warning($"[Z-Image] Text-encoder cleanup also failed while propagating " +
                        $"'{phaseError.Message}': {cleanupError.Message}");
                }
            }

            // Transactional cache replacement: neither prior slot is touched until every required encode and the
            // phase cleanup have succeeded. A failed negative encode therefore cannot destroy a valid positive hit.
            if (newPositive is not null)
            {
                _cachedPositive?.Dispose();
                _cachedPositive = newPositive;
                _cachedPositiveKey = newPositiveKey;
                newPositive = null;
            }
            if (newNegative is not null)
            {
                _cachedNegative?.Dispose();
                _cachedNegative = newNegative;
                _cachedNegativeKey = newNegativeKey;
                newNegative = null;
            }
            return regionalPlan;
        }
        catch
        {
            if (regionalPlan is not null)
            {
                RegionalPromptResolver.DisposeRegions(regionalPlan);
            }
            else if (regionalCandidates is not null)
            {
                // Resolve can fail after one or more callback results were produced but before returning a plan.
                foreach (Tensor candidate in regionalCandidates)
                    candidate.Dispose();
            }
            newPositive?.Dispose();
            newNegative?.Dispose();
            throw;
        }
    }

    private Tensor EncodePrompt(int[] tokenIds, int layerIndex)
    {
        if (tokenIds.Length <= 0)
            throw new InvalidOperationException("Z-Image's chat template produced no real prompt tokens.");

        // With batch one and causal attention, right-padding cannot affect any real-token hidden state. Avoid
        // running Qwen over hundreds of future pad positions, and return the already correctly sized output rather
        // than forcing a device-to-host slice after the encoder.
        return _qwen.EncodeMultiLayer(_textBackend, new[] { tokenIds }, new[] { layerIndex });
    }

    private Exception? CleanupTextEncoderPhase()
    {
        Exception? first = null;
        Try(() => _textBackend.Sync());
        // Qwen's fused-attention plans can retain several GB of per-shape workspace after its weights are gone.
        // Conditioning is cached on the host, so keeping those plans buys nothing on a repeat prompt and can starve
        // the following DiT/VAE phase (including when TE and VAE share a non-primary placement device).
        Try(() => _textBackend.ReleaseAttentionExecutionCache());
        Try(() => _textBackend.FreeWeights(_qwen.EnumerateWeights()));
        Try(() => _textBackend.FreeActivations(trimPool: true));
        return first;

        void Try(Action cleanup)
        {
            try
            {
                cleanup();
            }
            catch (Exception ex)
            {
                first ??= ex;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        void TryDispose(string resource, Action cleanup)
        {
            try
            {
                cleanup();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Z-Image] Failed to release {resource} during disposal: {ex.Message}");
            }
        }

        if (_cachedPositive is not null)
            TryDispose("cached positive embedding", _cachedPositive.Dispose);
        _cachedPositive = null;
        _cachedPositiveKey = null;
        if (_cachedNegative is not null)
            TryDispose("cached negative embedding", _cachedNegative.Dispose);
        _cachedNegative = null;
        _cachedNegativeKey = null;

        // The diffusion pipeline releases its DiT/VAE device state but deliberately does not own the injected
        // component objects. This wrapper is their recipe-level owner and tears them down before the mmap loaders.
        TryDispose("pipeline state", _pipeline.Dispose);
        TryDispose("text-encoder device weights", () => _textBackend.FreeWeights(_qwen.EnumerateWeights()));
        TryDispose("transformer device weights", () => _backend.FreeWeights(_transformer.EnumerateWeights()));
        TryDispose("VAE-decoder device weights", () => _pipeline.VaeBackend.FreeWeights(_vae.EnumerateWeights()));
        TryDispose("VAE-encoder device weights", () => _pipeline.VaeBackend.FreeWeights(_vaeEncoder.EnumerateWeights()));
        TryDispose("text encoder", _qwen.Dispose);
        TryDispose("transformer", _transformer.Dispose);
        TryDispose("tokenizer", _tokenizer.Dispose);
        foreach (Tensor weight in _qwenWeightTensors)
            TryDispose("text-encoder host weight", weight.Dispose);
        foreach (Tensor weight in _transformerWeightTensors)
            TryDispose("transformer host weight", weight.Dispose);
        foreach (Tensor weight in _ownedVaeWeights)
            TryDispose("VAE host weight", weight.Dispose);
        TryDispose("checkpoint loader", _checkpointLoader.Dispose);
        TryDispose("text-encoder loader", _qwenLoader.Dispose);
        TryDispose("VAE loader", _vaeLoader.Dispose);
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
