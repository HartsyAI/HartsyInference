using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Cuts the subject out of a finished generation. The matte (RMBG-1.4 unless
/// <see cref="ImageRequest.RemoveBackgroundModel"/> names RMBG 3.0) becomes <see cref="ImageResult.Alpha"/> and
/// the generated RGB is left exactly as it was, so the caller composites the cutout against whatever it likes. Mirrors
/// where the ComfyUI backend puts its <c>SwarmRemBg</c> node — last, over the final pixels, after every refine pass.
/// <para>Deliberately NOT an <see cref="Recipes.ImageFeatures"/> bit: this is pixel post-processing that works for any
/// family, so no recipe has to declare it.</para></summary>
public static class BackgroundRemovalStage
{
    /// <summary>The matter ids <see cref="ImageRequest.RemoveBackgroundModel"/> accepts; RMBG-1.4 is the default.</summary>
    public const string DefaultModelId = "rmbg";

    /// <summary>The catalog id of the matter <paramref name="request"/> asks for, or null when it asks for no cutout.</summary>
    /// <exception cref="NotSupportedException">The model is not a matter this stage runs, or is set without <see cref="ImageRequest.RemoveBackground"/>.</exception>
    public static string? ModelIdFor(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        bool named = !string.IsNullOrWhiteSpace(request.RemoveBackgroundModel);
        if (request.RemoveBackground != true)
        {
            return named ? throw new NotSupportedException("RemoveBackgroundModel needs RemoveBackground to be on.") : null;
        }
        string id = named ? request.RemoveBackgroundModel!.Trim().ToLowerInvariant() : DefaultModelId;
        return id == DefaultModelId || HasFeature(id, CatalogFeatures.BackgroundRemoval)
            ? id
            : throw new NotSupportedException($"Background removal runs rmbg (RMBG-1.4) or a catalog model flagged BackgroundRemoval; '{id}' is not one of them.");
    }

    /// <summary>Whether the cutout runs a model that needs the generating pipeline released first (catalog flag).</summary>
    public static bool NeedsGeneratorReleased(ImageRequest request) =>
        ModelIdFor(request) is string id && HasFeature(id, CatalogFeatures.ReleasesGeneratorFirst);

    private static bool HasFeature(string id, CatalogFeatures feature) =>
        ModelCatalog.Find(id)?.Features.HasFlag(feature) == true;

    /// <summary>Returns <paramref name="result"/> carrying an alpha plane when the request asked for a cutout, else
    /// unchanged. The RGB bytes are never rewritten — a partial-coverage edge must be composited once by the consumer,
    /// and pre-blending it here would darken the same pixels twice.</summary>
    public static ImageResult Apply(InferenceEngine engine, ImageRequest request, ImageResult result, CancellationToken cancel = default)
    {
        string? modelId = ModelIdFor(request);
        if (modelId is null)
        {
            return result;
        }
        ModelSpec spec = ModelResolver.Resolve(modelId, null, Modality.Vision);
        ImageData image = new ImageData { Rgb = result.Rgb, Width = result.Width, Height = result.Height };
        byte[] alpha = Rmbg3Matte.Handles(spec)
            ? Rmbg3Matte.AlphaAsync(engine, spec, image, cancel).GetAwaiter().GetResult()
            : engine.VisionInternal.Matte(spec, image);
        Logs.Verbose($"[Features][RemoveBackground] Matted {result.Width}x{result.Height} with {modelId}.");
        return result with { Alpha = alpha };
    }
}
