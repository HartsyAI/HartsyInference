using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Turns an <see cref="Inpaint"/> request into the pixel-space mask <c>[1, 1, H, W]</c> F32 in <c>[0, 1]</c> the
/// inpaint pipelines consume: resize to the target, grow (max-filter dilation), blur. Mask convention matches the host
/// UX — white (1.0) is inpainted, black (0.0) preserved, gray blends. Caller owns the returned tensor.
///
/// <para><see cref="Inpaint.ShrinkGrow"/> ("inpaint only masked") is handled a layer up by
/// <see cref="InpaintOnlyMasked"/>, which crops before generation and composites after; by the time a request reaches
/// this resolver the field has been cleared, so a non-zero value here is a routing bug rather than a user error.</para></summary>
public static class MaskResolver
{
    /// <summary>Builds the mask tensor for <paramref name="inpaint"/>, or null when there is no mask. Caller disposes.</summary>
    public static Tensor? Resolve(Inpaint? inpaint, int targetWidth, int targetHeight)
    {
        if (inpaint?.Mask is null)
        {
            return null;
        }
        byte[] maskBytes = FeatureImaging.ResizeGrayscale(inpaint.Mask, targetWidth, targetHeight);
        if (inpaint.Grow > 0)
        {
            FeatureImaging.DilateInPlace(maskBytes, targetWidth, targetHeight, inpaint.Grow);
        }
        if (inpaint.Blur > 0)
        {
            // "Mask Blur" is loosely a kernel size in pixels; sigma = blur / 2 puts the half-power radius at the user's intent.
            FeatureImaging.GaussianBlurInPlace(maskBytes, targetWidth, targetHeight, inpaint.Blur / 2.0f);
        }
        if (inpaint.ShrinkGrow != 0)
        {
            throw new InvalidOperationException(
                $"Inpaint.ShrinkGrow ({inpaint.ShrinkGrow}) reached the mask resolver still set. 'Inpaint only masked' is "
                + $"applied by {nameof(InpaintOnlyMasked)} above the pipeline, which clears the field once it has cropped — "
                + "seeing it here means a caller drove a recipe pipeline directly and would silently get a full-canvas "
                + "inpaint instead. Route through IImagesService.GenerateAsync.");
        }
        Tensor mask = FeatureImaging.GrayToMaskTensor(maskBytes, targetWidth, targetHeight);
        Logs.Verbose($"[Features][Mask] enabled: {targetWidth}x{targetHeight}, grow={inpaint.Grow}px, blur={inpaint.Blur}px.");
        return mask;
    }
}
