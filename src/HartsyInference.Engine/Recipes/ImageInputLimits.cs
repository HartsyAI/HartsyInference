using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>How many input images one image family can actually consume in a single request, and whether its reference
/// images only mean anything next to an init image. Declared per recipe through
/// <see cref="IArchitectureRecipe.InputLimits"/> so a host can size its upload UI from the same numbers the feature gate
/// refuses on, instead of guessing and finding out after the generation ran.</summary>
/// <remarks>The count covers the init image plus <see cref="ImageRequest.ReferenceImages"/>. IP-Adapter prompt images and
/// ControlNet hint images are separate composition objects with their own gates and are not counted here.</remarks>
/// <param name="MaxImages">Most images (init + references) the family reads. Zero for a text-to-image-only family.</param>
/// <param name="ReferencesRequireInitImage">True when the family edits exactly the init image and never reads
/// <see cref="ImageRequest.ReferenceImages"/> on their own, so references with no init image would silently fall back to
/// text-to-image.</param>
public readonly record struct ImageInputLimits(int MaxImages, bool ReferencesRequireInitImage)
{
    /// <summary>No image input at all: the family is text-to-image only.</summary>
    public static ImageInputLimits TextOnly => new ImageInputLimits(0, false);

    /// <summary>One init image, nothing else. Every denoise family, and every reference-edit family that reads only the init image.</summary>
    public static ImageInputLimits SingleInitImage => new ImageInputLimits(1, true);

    /// <summary>The limits implied by a family's declared features when it declares none of its own: any init-image
    /// feature means exactly one image, otherwise none.</summary>
    public static ImageInputLimits DerivedFrom(ImageFeatures supports) =>
        (supports & (ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit)) != 0 ? SingleInitImage : TextOnly;

    /// <summary>Number of images <paramref name="request"/> hands the family: the init image plus every non-null reference.</summary>
    internal static int CountImages(ImageRequest request)
    {
        int count = request.Img2Img?.InitImage is not null ? 1 : 0;
        if (request.ReferenceImages is not null)
        {
            foreach (ImageData reference in request.ReferenceImages)
            {
                if (reference is not null)
                {
                    count++;
                }
            }
        }
        return count;
    }

    /// <summary>The refusal message for a request these limits cannot honour, or null when it fits. Kept short on
    /// purpose: hosts show it verbatim in chat cards.</summary>
    /// <remarks>A missing init image is reported before an over-count. When both are true - references and no init
    /// image on a family that needs one - "add an init image" is the fix, and "send fewer images" would only lead
    /// the caller to drop references until they hit the same wall.</remarks>
    internal string? Violation(string familyId, ImageRequest request)
    {
        if (ReferencesRequireInitImage && request.Img2Img?.InitImage is null && request.ReferenceImages is { Count: > 0 })
        {
            return $"Model family '{familyId}' needs an init image to edit; reference images alone are not used.";
        }
        int images = CountImages(request);
        if (images > MaxImages)
        {
            return $"Model family '{familyId}' takes at most {MaxImages} input image{(MaxImages == 1 ? "" : "s")}; {images} were supplied.";
        }
        return null;
    }
}
