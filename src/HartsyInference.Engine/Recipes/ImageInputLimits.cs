using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>How many input images a family reads per request, and whether references need an init image.</summary>
/// <remarks>The count covers the init image plus <see cref="ImageRequest.ReferenceImages"/>. IP-Adapter prompt images and
/// ControlNet hint images are separate composition objects with their own gates and are not counted here.</remarks>
/// <param name="MaxImages">Most images (init + references) the family reads. Zero for a text-to-image-only family.</param>
/// <param name="ReferencesRequireInitImage">True when the family reads only the init image.</param>
/// <param name="MaxReferences">Most images the family conditions on as references, counting an init image sent in
/// <see cref="Img2ImgMode.Reference"/>; null when <paramref name="MaxImages"/> is the only cap. A family that also
/// denoises an init image reads one image more than it conditions on, which <paramref name="MaxImages"/> alone cannot
/// say.</param>
public readonly record struct ImageInputLimits(int MaxImages, bool ReferencesRequireInitImage, int? MaxReferences = null)
{
    /// <summary>No image input at all: the family is text-to-image only.</summary>
    public static ImageInputLimits TextOnly => new ImageInputLimits(0, false);

    /// <summary>One init image and nothing else: every denoise family and every init-only edit family.</summary>
    public static ImageInputLimits SingleInitImage => new ImageInputLimits(1, true);

    /// <summary>The limits a family's features imply: one image for any init-image feature, otherwise none.</summary>
    public static ImageInputLimits DerivedFrom(ImageFeatures supports) =>
        (supports & (ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit)) != 0 ? SingleInitImage : TextOnly;

    /// <summary>Number of images <paramref name="request"/> hands the family: the init image plus every non-null reference.</summary>
    internal static int CountImages(ImageRequest request) =>
        (request.Img2Img?.InitImage is not null ? 1 : 0) + CountReferenceImages(request);

    /// <summary>Number of images <paramref name="request"/> asks the family to condition on: every non-null reference,
    /// plus the init image when it is sent in <see cref="Img2ImgMode.Reference"/>.</summary>
    internal static int CountReferences(ImageRequest request) =>
        (request.Img2Img is { Mode: Img2ImgMode.Reference, InitImage: not null } ? 1 : 0) + CountReferenceImages(request);

    private static int CountReferenceImages(ImageRequest request)
    {
        int count = 0;
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

    /// <summary>A short refusal for a request past these limits, or null; hosts show it verbatim.</summary>
    /// <remarks>A missing init image is reported first: adding one is the fix, sending fewer is not.</remarks>
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
        int references = CountReferences(request);
        if (MaxReferences is int maxReferences && references > maxReferences)
        {
            return $"Model family '{familyId}' conditions on at most {maxReferences} reference image{(maxReferences == 1 ? "" : "s")}; "
                + $"{references} were supplied (an init image in Reference mode counts as one).";
        }
        return null;
    }
}
