using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2's input and output conversions: upstream <c>to_normalized_tensor</c> (RGB24 to ImageNet-normalised <c>[1, 3, H, W]</c> at the native size, no
/// resize; <see cref="DepthPro2Network"/> replicate-pads to a multiple of 32 and crops back) and the metric step of <c>infer</c>.
/// </summary>
public static class DepthPro2Preprocessor
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    /// <summary>Interleaved RGB24 as <c>(x / 255 - mean) / std</c> in a new F32 <c>[1, 3, H, W]</c> tensor the caller owns.</summary>
    public static Tensor Normalize(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (width <= 0 || height <= 0 || rgb.Length != (long)width * height * 3)
            throw new ArgumentException($"Expected {width}x{height}x3 RGB bytes; got {rgb.Length}.", nameof(rgb));
        Tensor pixels = new(new TensorShape(1, 3, height, width), DType.F32);
        Span<float> o = pixels.AsSpan<float>();
        int plane = width * height;
        for (int c = 0; c < 3; c++)
        {
            float mean = Mean[c], std = Std[c];
            Span<float> channel = o.Slice(c * plane, plane);
            for (int i = 0; i < plane; i++)
                channel[i] = (rgb[i * 3 + c] / 255f - mean) / std;
        }

        return pixels;
    }

    /// <summary>
    /// Metric depth in metres along the optical axis, <c>f_px / (W · inverseDepth)</c> in float32 as upstream computes it, with <paramref name="width"/> the
    /// unpadded image width. Where the inverse depth is not above zero (sky, infinitely far) the depth is positive infinity, as upstream leaves it, and the
    /// mask is 0; elsewhere the mask is 1.
    /// </summary>
    public static void ToMetric(ReadOnlySpan<float> inverseDepth, int width, double focalLengthPx, Span<float> depthMeters, Span<float> validMask)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "The width must be positive.");
        if (!double.IsFinite(focalLengthPx) || focalLengthPx <= 0)
            throw new ArgumentOutOfRangeException(nameof(focalLengthPx), $"The focal length must be a positive number of pixels; got {focalLengthPx}.");
        if (depthMeters.Length != inverseDepth.Length || validMask.Length != inverseDepth.Length)
            throw new ArgumentException("The depth and mask spans must match the inverse depth's length.");
        float focal = (float)focalLengthPx;
        for (int i = 0; i < inverseDepth.Length; i++)
        {
            float inverse = inverseDepth[i];
            bool valid = inverse > 0f;
            depthMeters[i] = valid ? focal / (width * inverse) : float.PositiveInfinity;
            validMask[i] = valid ? 1f : 0f;
        }
    }
}
