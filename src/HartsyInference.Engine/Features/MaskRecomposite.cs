using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Pastes a full-canvas masked result over the init image through the mask, hard-thresholded like SwarmUI's CompositeMask.
/// Applies to any masked request, including one whose family used the init as a reference.</summary>
public static class MaskRecomposite
{
    /// <summary>Composites <paramref name="generated"/> over the init image; unchanged without a mask or with recomposite off.</summary>
    public static ImageResult Apply(ImageRequest request, ImageResult generated)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(generated);
        if (request.Inpaint?.Mask is null || request.Img2Img is null || !request.Inpaint.RecompositeMask)
        {
            return generated;
        }

        // Sized to the pipeline's actual output, which a family that snaps dimensions may have changed.
        byte[] mask = MaskResolver.ResolveBytes(request.Inpaint, generated.Width, generated.Height)!;
        FeatureImaging.ApplyCompositePolicy(mask, request.MaskCompositeUnthresholded);
        ImageData init = request.Img2Img.InitImage;
        ImageData original = init.Width == generated.Width && init.Height == generated.Height
            ? init
            : new ImageData
            {
                Rgb = FeatureImaging.ResizeRgb24(init, generated.Width, generated.Height),
                Width = generated.Width,
                Height = generated.Height,
            };
        ImageData patch = new ImageData { Rgb = generated.Rgb, Width = generated.Width, Height = generated.Height };
        ImageData composited = FeatureImaging.CompositeRgb24(original, patch, mask, 0, 0);
        return generated with { Rgb = composited.Rgb };
    }
}
