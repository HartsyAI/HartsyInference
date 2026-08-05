using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Everything the UNet-family pipelines (SD 1.5, SDXL) need resolved for one generation: ControlNet layers,
/// IP-Adapter conditioning, the img2img/inpaint init, the variation-seed noise, and any weighted-prompt conditioning
/// schedule. Built once per request and disposed after the pipeline call — every tensor here is owned by the plan.</summary>
public sealed class UnetCompositionPlan : IDisposable
{
    /// <summary>Resolved ControlNet layers, or null when the request set none.</summary>
    public ControlNetResolver.ResolvedSpec? ControlNets { get; private init; }

    /// <summary>Resolved IP-Adapter conditioning, or null when no adapter model was named.</summary>
    public IpAdapterResolver.ResolvedSpec? IpAdapters { get; private init; }

    /// <summary>Resolved img2img/inpaint init, or null for pure text-to-image.</summary>
    public Img2ImgResolver.Img2ImgSpec? Img2Img { get; private init; }

    private Tensor? _variationNoise;

    /// <summary>Pre-blended variation-seed noise for the text-to-image path, or null to seed normally.</summary>
    public Tensor? VariationNoise
    {
        get => _variationNoise;
        private init => _variationNoise = value;
    }

    /// <summary>Hands the variation noise to the pipeline, which takes ownership and disposes it; the plan stops
    /// tracking it so it is not double-freed.</summary>
    public Tensor? TakeVariationNoise()
    {
        Tensor? noise = _variationNoise;
        _variationNoise = null;
        return noise;
    }

    /// <summary>Weighted-prompt conditioning schedule, or null when the prompts carry no weighting syntax.</summary>
    public ConditioningSchedule? Conditioning { get; private init; }

    /// <summary>Resolves every composition object on <paramref name="request"/> against a UNet-family pipeline.
    /// <paramref name="conditioningFactory"/> supplies the architecture's weighted-conditioning build (single- or
    /// dual-CLIP); the IP-Adapter cache callbacks are owned by the calling pipeline.</summary>
    public static UnetCompositionPlan Build(
        ImageRequest request,
        IBackend backend,
        UNetConfig unetConfig,
        IpAdapterBaseModel baseModel,
        Func<ConditioningSchedule?> conditioningFactory,
        Func<string, IpAdapterCacheEntry?> cacheLookup,
        Action<IpAdapterCacheEntry> cachePut,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(conditioningFactory);
        (int width, int height) = RecipeRequestMapper.Size(request);
        ControlNetResolver.ResolvedSpec? controlNets = null;
        IpAdapterResolver.ResolvedSpec? ipAdapters = null;
        Img2ImgResolver.Img2ImgSpec? img2img = null;
        Tensor? variationNoise = null;
        ConditioningSchedule? conditioning = null;
        try
        {
            // Resolved first so the inpaint-without-init-image guard fires before the IP-Adapter resolver loads a model.
            img2img = RecipeImg2ImgBinder.Resolve(request, width, height);

            controlNets = ControlNetResolver.Resolve(
                request.ControlNets, unetConfig, width, height,
                static message => Logs.Info($"[Features][ControlNet] {message}"),
                index => RequestExtras.ControlNetUnionType(request.Extra, index));

            string? adapterModel = RequestExtras.String(request.Extra, RequestExtras.IpAdapterModel);
            if (request.IpAdapter is not null && string.IsNullOrWhiteSpace(adapterModel))
            {
                throw new InvalidOperationException(
                    $"Image-prompt conditioning was requested but no adapter was named. Set Extra[\"{RequestExtras.IpAdapterModel}\"] to the IP-Adapter model.");
            }
            ipAdapters = IpAdapterResolver.ResolveAsync(
                request.IpAdapter,
                adapterModel,
                RequestExtras.String(request.Extra, RequestExtras.IpAdapterClipVision),
                RequestExtras.Number(request.Extra, RequestExtras.IpAdapterWeight, 1.0),
                RequestExtras.Number(request.Extra, RequestExtras.IpAdapterStart, 0.0),
                RequestExtras.Number(request.Extra, RequestExtras.IpAdapterEnd, 1.0),
                RequestExtras.String(request.Extra, RequestExtras.IpAdapterWeightType, "standard") ?? "standard",
                backend,
                baseModel,
                static message => Logs.Info($"[Features][IPAdapter] {message}"),
                cacheLookup,
                cachePut,
                cancel).GetAwaiter().GetResult();
            if (ipAdapters?.FaceIdLoraPath is not null)
            {
                Logs.Warning(
                    "[Features][IPAdapter] This FaceID adapter ships a companion UNet LoRA that is NOT applied — LoRA is merged "
                    + "at model construction, so add it to the request's LoRA stack for full FaceID fidelity.");
            }

            if (img2img is null)
            {
                variationNoise = VariationSeedResolver.Resolve(
                    request.VariationSeed, width, height, request.Seed, VariationSeedResolver.SdLatentChannels);
            }
            else if (request.VariationSeed is not null)
            {
                Logs.Warning("[Features][Variation] Variation seed is ignored on the img2img path — the init latent replaces the seeded noise.");
            }

            conditioning = conditioningFactory();
            return new UnetCompositionPlan
            {
                ControlNets = controlNets,
                IpAdapters = ipAdapters,
                Img2Img = img2img,
                VariationNoise = variationNoise,
                Conditioning = conditioning,
            };
        }
        catch (Exception ex)
        {
            Logs.Error("[Features] Composition resolution failed; rolling back partial state.", ex);
            DisposeParts(controlNets, ipAdapters, img2img, variationNoise, conditioning);
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeParts(ControlNets, IpAdapters, Img2Img, TakeVariationNoise(), Conditioning);

    /// <summary>Disposes whichever parts were built; safe on a partially-constructed plan.</summary>
    private static void DisposeParts(
        ControlNetResolver.ResolvedSpec? controlNets,
        IpAdapterResolver.ResolvedSpec? ipAdapters,
        Img2ImgResolver.Img2ImgSpec? img2img,
        Tensor? variationNoise,
        ConditioningSchedule? conditioning)
    {
        controlNets?.Dispose();
        ipAdapters?.Dispose();
        img2img?.Dispose();
        variationNoise?.Dispose();
        if (conditioning is not null)
        {
            foreach (Tensor variant in conditioning.Variants)
            {
                variant.Dispose();
            }
        }
    }
}
