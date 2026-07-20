using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Turns an <see cref="Img2Img"/> (and, when present, an <see cref="Inpaint"/>) request into the source tensor +
/// strength + optional mask a pipeline needs to build an image-to-image request. The init image is resized to the request
/// resolution because the pipelines require source dims == request dims.
///
/// <para>With a mask attached the pipeline runs blend-on-vanilla inpaint, which works with any vanilla checkpoint — no
/// dedicated 9-channel inpaint UNet required.</para></summary>
public static class Img2ImgResolver
{
    /// <summary>Resolved img2img inputs; owns both tensors.</summary>
    public sealed class Img2ImgSpec : IDisposable
    {
        /// <summary>Source tensor <c>[1, 3, H, W]</c> in <c>[-1, 1]</c>.</summary>
        public required Tensor SourceTensor { get; init; }

        /// <summary>Denoise strength in <c>[0, 1]</c>.</summary>
        public required float Strength { get; init; }

        /// <summary>Optional inpaint mask <c>[1, 1, H, W]</c> F32 in <c>[0, 1]</c> (1 = inpaint, 0 = preserve); null for plain img2img.</summary>
        public Tensor? MaskTensor { get; init; }

        /// <summary>Frees the source and mask tensors.</summary>
        public void Dispose()
        {
            SourceTensor?.Dispose();
            MaskTensor?.Dispose();
        }
    }

    /// <summary>Builds the spec, or null when there is no init image. Caller disposes the returned spec.
    /// Strength 0 is a valid (degenerate) pass-through state and is not filtered out.</summary>
    public static Img2ImgSpec? Resolve(Img2Img? img2img, Inpaint? inpaint, int targetWidth, int targetHeight)
    {
        if (img2img?.InitImage is null)
        {
            return null;
        }
        float strength = (float)Math.Clamp(img2img.Creativity, 0.0, 1.0);
        byte[] rgb = FeatureImaging.ResizeRgb24(img2img.InitImage, targetWidth, targetHeight);
        Tensor tensor = FeatureImaging.RgbToTensorMinusOneOne(rgb, targetWidth, targetHeight);
        Tensor? mask = null;
        try
        {
            mask = MaskResolver.Resolve(inpaint, targetWidth, targetHeight);
        }
        catch (Exception ex)
        {
            Logs.Error("[Features][Img2img] Mask resolution failed.", ex);
            tensor.Dispose();
            throw;
        }
        Logs.Verbose($"[Features][Img2img] Enabled: target={targetWidth}x{targetHeight}, strength={strength:F2}, mode={(mask is null ? "img2img" : "inpaint")}.");
        return new Img2ImgSpec
        {
            SourceTensor = tensor,
            Strength = strength,
            MaskTensor = mask,
        };
    }
}
