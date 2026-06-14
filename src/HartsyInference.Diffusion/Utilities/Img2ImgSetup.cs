using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>The image-to-image / inpaint prelude that every pipeline does identically: validate source shape, validate optional mask, clamp strength, compute the starting denoise step, and detect the strength=0 short-circuit. Pipelines call this once near the top of <c>GenerateFromTokens</c> and then act on the returned <see cref="Plan"/>.</summary>
public static class Img2ImgSetup
{
    /// <summary>The validated img2img inputs and the derived denoise-loop start step. <see cref="PassThrough"/> is true when the caller should short-circuit (strength clamped to 0 → 0 denoise steps requested), in which case the source image should be returned to the user unchanged.</summary>
    public readonly record struct Plan(
        int StartStep,
        Tensor? MaskPixel,
        bool PassThrough);

    /// <summary>The no-op plan returned for plain text-to-image requests. Denoise loop starts at step 0; no mask; not a pass-through.</summary>
    public static Plan TextToImage => new Plan(StartStep: 0, MaskPixel: null, PassThrough: false);

    /// <summary>Validates an <see cref="ImageToImageRequest"/> against the pipeline's expected image dimensions and computes the denoise loop's start step. The <paramref name="height"/> / <paramref name="width"/> are the request's image size; source and mask must match exactly (no resize is applied here — callers that want resize should do it before invoking the pipeline).
    /// <para>Returns <see cref="TextToImage"/> immediately for plain <see cref="TextToImageRequest"/>. Returns a plan with <c>PassThrough=true</c> when <c>strength</c> clamps to 0 (no denoise needed; caller should byte-copy the source through <see cref="ImagePostProcessor.TensorToRgbBytes"/>).</para>
    /// </summary>
    public static Plan Prepare(TextToImageRequest request, int height, int width, int steps)
    {
        if (request is not ImageToImageRequest img2img) return TextToImage;

        Tensor src = img2img.SourceImage;
        if (src.Shape.Rank != 4 || src.Shape[0] != 1 || src.Shape[1] != 3 ||
            src.Shape[2] != height || src.Shape[3] != width)
        {
            throw new ArgumentException(
                $"SourceImage shape must be [1, 3, {height}, {width}] (matching request); got {src.Shape}.",
                nameof(request));
        }

        Tensor? mask = null;
        if (img2img.Mask is not null)
        {
            Tensor mk = img2img.Mask;
            if (mk.Shape.Rank != 4 || mk.Shape[0] != 1 || mk.Shape[1] != 1 ||
                mk.Shape[2] != height || mk.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"Mask shape must be [1, 1, {height}, {width}] (matching source); got {mk.Shape}.",
                    nameof(request));
            }
            if (mk.DType != DType.F32)
            {
                throw new ArgumentException($"Mask must be F32 in [0, 1]; got {mk.DType}.", nameof(request));
            }
            mask = mk;
        }

        float strength = Math.Clamp(img2img.Strength, 0f, 1f);
        int initTimesteps = (int)MathF.Round(steps * strength);
        int startStep = Math.Max(steps - initTimesteps, 0);
        return new Plan(startStep, mask, PassThrough: initTimesteps == 0);
    }
}
