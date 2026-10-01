using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Pastes a full-canvas masked result over the init image through the mask, hard-thresholded like SwarmUI's CompositeMask.</summary>
public static class MaskRecomposite
{
    /// <summary>Smallest nonzero mask value; SwarmUI's 0.001 threshold falls below it.</summary>
    private const byte Threshold = 1;

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
        if (request.MaskCompositeUnthresholded)
        {
            FeatureImaging.ThresholdInPlace(mask, Threshold);
        }
        else
        {
            FeatureImaging.BinarizeInPlace(mask, Threshold);
        }
        ImageData original = new ImageData
        {
            Rgb = FeatureImaging.ResizeRgb24(request.Img2Img.InitImage, generated.Width, generated.Height),
            Width = generated.Width,
            Height = generated.Height,
        };
        ImageData patch = new ImageData { Rgb = generated.Rgb, Width = generated.Width, Height = generated.Height };
        ImageData composited = FeatureImaging.CompositeRgb24(original, patch, mask, 0, 0);
        return generated with { Rgb = composited.Rgb };
    }
}
