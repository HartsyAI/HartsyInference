using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Binds an <see cref="ImageRequest"/>'s img2img/inpaint composition onto the diffusion package's request
/// record, so every architecture recipe reaches the pipeline's image-to-image path through one implementation instead
/// of a per-family copy of the same mapping.
/// <para>Deliberately narrower than <see cref="UnetCompositionPlan"/>: that type also resolves ControlNet, IP-Adapter
/// and variation-seed noise, all of which are parameterized by UNet-only types (<c>UNetConfig</c>,
/// <c>IpAdapterBaseModel</c>, SD latent channel counts). Img2img is the one composition object every family — UNet,
/// DiT and pixel-space alike — expresses identically, so it is the only one bound here.</para></summary>
public static class RecipeImg2ImgBinder
{
    /// <summary>Resolves the init image (and inpaint mask) at <paramref name="width"/> x <paramref name="height"/>, or
    /// null for pure text-to-image. Caller owns the returned spec and must dispose it after the pipeline call.
    /// <para><b>Pass the pipeline's own resolved size, not the raw request size.</b> The resolver resizes the init
    /// image to exactly these dimensions and <c>Img2ImgSetup.Prepare</c> later compares the source shape against the
    /// size the pipeline computed for itself — so a family that snaps or clamps its dimensions (Krea 2 and Mage-Flow
    /// to a multiple of 16, Flux.2 to the VAE downscale factor, Chroma-Radiance to the patch grid) must resolve at the
    /// snapped size and write that same size into the request it hands the pipeline. Passing the unsnapped size
    /// surfaces as an <c>ArgumentException</c> at generation time.</para></summary>
    public static Img2ImgResolver.Img2ImgSpec? Resolve(ImageRequest request, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Inpaint is not null && request.Img2Img is null)
        {
            throw new InvalidOperationException(
                "An inpaint mask was supplied without an init image. Inpainting re-paints an existing image — set Img2Img.InitImage too.");
        }
        return Img2ImgResolver.Resolve(request.Img2Img, request.Inpaint, width, height);
    }

    /// <summary>Promotes <paramref name="inner"/> to an <see cref="ImageToImageRequest"/> carrying the resolved source,
    /// strength and mask, or returns it unchanged when <paramref name="spec"/> is null. Every field the family already
    /// set on <paramref name="inner"/> (scheduler, CLIP skip, sigma shift, …) is carried over by the record copy
    /// constructor, so a new base-request field cannot be silently dropped on the img2img path.
    /// <para>The returned request borrows the spec's tensors — the spec must outlive the pipeline call.</para></summary>
    /// <remarks>Variation-seed noise is meaningless once an init latent replaces the seeded noise, so callers should
    /// resolve <c>InitialNoise</c> only when this method's <paramref name="spec"/> is null.</remarks>
    public static TextToImageRequest Apply(TextToImageRequest inner, Img2ImgResolver.Img2ImgSpec? spec)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (spec is null)
        {
            return inner;
        }
        return new ImageToImageRequest(inner, spec.SourceTensor)
        {
            Strength = spec.Strength,
            Mask = spec.MaskTensor,
        };
    }
}
