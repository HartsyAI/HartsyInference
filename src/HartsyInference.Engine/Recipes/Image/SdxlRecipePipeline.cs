using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SDXL pipeline driven against the native <see cref="ImageRequest"/>. Encodes the prompt/negative
/// through the embedded CLIP-L/CLIP-G vocab, resolves the request's composition objects through
/// <see cref="UnetCompositionPlan"/> plus the refiner swap, and runs <see cref="SdxlPipeline.GenerateFromTokens"/>.</summary>
public sealed class SdxlRecipePipeline : IRecipePipeline
{
    private readonly SdxlPipeline _pipeline;
    private readonly ClipTokenizer _tokenizer = new ClipTokenizer();
    private readonly IBackend _backend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly MergedLoraStack? _loraStack;
    private readonly Dictionary<string, IpAdapterCacheEntry> _ipAdapterCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, SdxlRefinerEntry> _refinerCache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>SDXL's dual-CLIP conditioning is penultimate-layer by spec, so weighted encoding uses the same depth.</summary>
    private const int SdxlLayersFromEnd = 2;

    /// <summary>Wraps the constructed SDXL pipeline plus its text encoders and merged LoRA stack, taking ownership of every disposable.</summary>
    public SdxlRecipePipeline(
        SdxlPipeline pipeline,
        IBackend backend,
        ClipTextEncoder clipL,
        ClipTextEncoder clipG,
        MergedLoraStack? loraStack)
    {
        _pipeline = pipeline;
        _backend = backend;
        _clipL = clipL;
        _clipG = clipG;
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        string negative = request.NegativePrompt ?? "";

        int[] tokensL = _tokenizer.Encode(request.Prompt);
        int[] negL = _tokenizer.Encode(negative);
        int[] tokensG = _tokenizer.Encode(request.Prompt);
        int[] negG = _tokenizer.Encode(negative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        using UnetCompositionPlan plan = UnetCompositionPlan.Build(
            request,
            _backend,
            UNetConfig.SdxlBase,
            IpAdapterBaseModel.Sdxl,
            () => WeightedConditioning.BuildDualClip(_backend, _clipL, _clipG, _tokenizer, request.Prompt, negative, SdxlLayersFromEnd),
            LookupIpAdapter,
            CacheIpAdapter,
            cancel);

        RefinerSwapConfig? refiner = ResolveRefiner(request);
        TextToImageRequest inner = BuildInner(request, negative, plan);

        int totalSteps = request.Steps ?? SdxlRecipe.FamilyDefaults.Steps;
        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            tokensL, negL, tokensG, negG, eosG, negEosG, inner,
            p => progress?.Report(new StepPreview { Step = p.Step, TotalSteps = totalSteps }),
            controlNets: plan.ControlNets?.Conditionings,
            refiner: refiner,
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
                ["arch"] = "sdxl",
                ["size"] = $"{width}x{height}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = totalSteps.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Builds the inner diffusion request — an <see cref="ImageToImageRequest"/> when the plan resolved an init image, else plain text-to-image with any variation-seed noise injected.</summary>
    private static TextToImageRequest BuildInner(ImageRequest request, string negative, UnetCompositionPlan plan)
    {
        if (plan.Img2Img is not null)
        {
            return new ImageToImageRequest
            {
                Prompt = request.Prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = request.Steps,
                CfgScale = request.CfgScale,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                Scheduler = request.Scheduler,
                SourceImage = plan.Img2Img.SourceTensor,
                Strength = plan.Img2Img.Strength,
                Mask = plan.Img2Img.MaskTensor,
            };
        }
        return new TextToImageRequest
        {
            Prompt = request.Prompt,
            NegativePrompt = negative,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps,
            CfgScale = request.CfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            Scheduler = request.Scheduler,
            InitialNoise = plan.TakeVariationNoise(),
        };
    }

    /// <summary>Resolves the refiner request into a mid-loop StepSwap config, loading (and caching) the refiner UNet.
    /// PostApply — the VAE-round-trip variant — is not wired; the request is served as StepSwap with a log note.</summary>
    private RefinerSwapConfig? ResolveRefiner(ImageRequest request)
    {
        RefinerResolver.RefinerSpec? spec = RefinerResolver.Resolve(request.Refiner, request.Steps ?? SdxlRecipe.FamilyDefaults.Steps, request.CfgScale ?? SdxlRecipe.FamilyDefaults.CfgScale);
        if (spec is null)
        {
            return null;
        }
        if (!string.Equals(spec.Method, "StepSwap", StringComparison.OrdinalIgnoreCase))
        {
            Logs.Info($"[SdxlRecipePipeline] Refiner method '{spec.Method}' is served as StepSwap (mid-loop UNet swap); PostApply is not wired.");
        }
        if (spec.Upscale is > 1f)
        {
            Logs.Warning($"[SdxlRecipePipeline] Refiner upscale {spec.Upscale} is ignored — StepSwap keeps the base latent resolution.");
        }
        if (!_refinerCache.TryGetValue(spec.Model, out SdxlRefinerEntry? entry))
        {
            entry = SdxlRefinerLoader.Load(spec.Model);
            _refinerCache[spec.Model] = entry;
        }
        return new RefinerSwapConfig { RefinerUnet = entry.Unet, Strength = spec.Strength };
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
        foreach (SdxlRefinerEntry entry in _refinerCache.Values)
        {
            entry.Dispose();
        }
        _refinerCache.Clear();
        _pipeline.Dispose();
        _tokenizer.Dispose();
        // The LoRA stack owns the merged tensors the components reference, so it outlives them by exactly this much.
        _loraStack?.Dispose();
        Logs.Verbose("[SdxlRecipePipeline] Disposed.");
    }
}
