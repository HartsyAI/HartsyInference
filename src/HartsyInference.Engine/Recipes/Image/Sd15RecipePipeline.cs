using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SD1.5 pipeline driven against the native <see cref="ImageRequest"/>. Encodes the prompt/negative through the embedded CLIP-L vocab, resolves the request's composition objects through <see cref="UnetCompositionPlan"/> (ControlNet / IP-Adapter / img2img / inpaint / variation seed / weighted prompts), and runs <see cref="StableDiffusion15Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Sd15Loader.Generate</c> drive path.</summary>
public sealed class Sd15RecipePipeline : IRecipePipeline
{
    private readonly StableDiffusion15Pipeline _pipeline;
    private readonly ClipTokenizer _tokenizer;
    private readonly SafeTensorsLoader _checkpointLoader;
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _textEncoder;
    private readonly MergedLoraStack? _loraStack;
    private readonly Dictionary<string, IpAdapterCacheEntry> _ipAdapterCache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Wraps the constructed SD1.5 pipeline plus its tokenizer, text encoder, and merged LoRA stack, taking ownership of every disposable.</summary>
    public Sd15RecipePipeline(
        StableDiffusion15Pipeline pipeline,
        ClipTokenizer tokenizer,
        SafeTensorsLoader checkpointLoader,
        IBackend backend,
        ClipTextEncoder textEncoder,
        MergedLoraStack? loraStack)
    {
        _pipeline = pipeline;
        _tokenizer = tokenizer;
        _checkpointLoader = checkpointLoader;
        _backend = backend;
        _textEncoder = textEncoder;
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        string negative = request.NegativePrompt ?? "";
        int layersFromEnd = RecipeRequestMapper.MapClipSkip(request.ClipSkip) ?? 1;

        int[] promptTokens = _tokenizer.Encode(request.Prompt);
        int[] negativeTokens = _tokenizer.Encode(negative);

        using UnetCompositionPlan plan = UnetCompositionPlan.Build(
            request,
            _backend,
            UNetConfig.Sd15,
            IpAdapterBaseModel.Sd15,
            () => WeightedConditioning.BuildSingleClip(_backend, _textEncoder, _tokenizer, request.Prompt, negative, layersFromEnd),
            LookupIpAdapter,
            CacheIpAdapter,
            cancel);

        TextToImageRequest inner = BuildInner(request, negative, plan);

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            promptTokens, negativeTokens, inner, bridge,
            controlNets: plan.ControlNets?.Conditionings,
            ipAdapters: plan.IpAdapters?.Conditionings,
            conditioningSchedule: plan.Conditioning);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "sd15",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = (request.Steps ?? Sd15Recipe.FamilyDefaults.Steps).ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Builds the inner diffusion request — an <see cref="ImageToImageRequest"/> when the plan resolved an init image, else plain text-to-image with any variation-seed noise injected.</summary>
    private static TextToImageRequest BuildInner(ImageRequest request, string negative, UnetCompositionPlan plan)
    {
        TextToImageRequest inner = RecipeRequestMapper.ToTextToImage(request, negative) with
        {
            Scheduler = request.Scheduler,
            ClipSkip = RecipeRequestMapper.MapClipSkip(request.ClipSkip),
            // The plan only resolves variation noise on the text-to-image path, so this is null under img2img.
            InitialNoise = plan.TakeVariationNoise(),
        };
        return RecipeImg2ImgBinder.Apply(inner, plan.Img2Img);
    }

    /// <summary>Cached IP-Adapter entry for <paramref name="path"/>, or null when not loaded yet.</summary>
    private IpAdapterCacheEntry? LookupIpAdapter(string path) =>
        _ipAdapterCache.TryGetValue(path, out IpAdapterCacheEntry? entry) ? entry : null;

    /// <summary>Stores a freshly loaded IP-Adapter entry for reuse across generations on this model.</summary>
    private void CacheIpAdapter(IpAdapterCacheEntry entry) => _ipAdapterCache[entry.FilePath] = entry;

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IpAdapterCacheEntry entry in _ipAdapterCache.Values)
        {
            entry.Dispose();
        }
        _ipAdapterCache.Clear();
        _pipeline.Dispose();
        _tokenizer.Dispose();
        // The LoRA stack owns the merged tensors the components reference, so it outlives them by exactly this much.
        _loraStack?.Dispose();
        _checkpointLoader.Dispose();
        Logs.Verbose("[Sd15RecipePipeline] Disposed.");
    }
}
