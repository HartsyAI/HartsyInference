using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Dinov2;

/// <summary>DINOv2 image preprocessing — square bilinear resize to <c>imageSize × imageSize</c> then
/// per-channel ImageNet normalization. Input is HWC-packed RGB u8; output is <c>[1, 3, size, size]</c> F32
/// ready for <see cref="Dinov2VisionEncoder"/>.</summary>
public sealed unsafe class Dinov2ImagePreprocessor
{
    private readonly int _imageSize;
    // ImageNet mean/std (the DINOv2 default).
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public int ImageSize => _imageSize;

    public Dinov2ImagePreprocessor(int imageSize)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "Image size must be positive.");
        _imageSize = imageSize;
    }

    /// <summary>Resizes + normalizes an interleaved-RGB24 image to <c>[1, 3, size, size]</c>.</summary>
    public Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int srcWidth, int srcHeight)
    {
        if (srcWidth <= 0 || srcHeight <= 0)
            throw new ArgumentException($"Source dimensions must be positive; got {srcWidth}x{srcHeight}.");
        int expected = srcWidth * srcHeight * 3;
        if (rgbPixels.Length != expected)
            throw new ArgumentException($"rgbPixels length {rgbPixels.Length} != expected {expected}.", nameof(rgbPixels));

        Tensor output = new(new TensorShape(1, 3, _imageSize, _imageSize), DType.F32);
        float* outPtr = (float*)output.DataPointer;
        long plane = (long)_imageSize * _imageSize;
        float scaleX = (float)srcWidth / _imageSize;
        float scaleY = (float)srcHeight / _imageSize;

        for (int oy = 0; oy < _imageSize; oy++)
        {
            float sy = (oy + 0.5f) * scaleY - 0.5f;
            int sy0 = (int)MathF.Floor(sy);
            float fy = sy - sy0;
            int y0 = Math.Clamp(sy0, 0, srcHeight - 1), y1 = Math.Clamp(sy0 + 1, 0, srcHeight - 1);
            float wy1 = fy, wy0 = 1f - fy;

            for (int ox = 0; ox < _imageSize; ox++)
            {
                float sx = (ox + 0.5f) * scaleX - 0.5f;
                int sx0 = (int)MathF.Floor(sx);
                float fx = sx - sx0;
                int x0 = Math.Clamp(sx0, 0, srcWidth - 1), x1 = Math.Clamp(sx0 + 1, 0, srcWidth - 1);
                float wx1 = fx, wx0 = 1f - fx;

                long dstPos = oy * _imageSize + ox;
                for (int c = 0; c < 3; c++)
                {
                    byte p00 = rgbPixels[(y0 * srcWidth + x0) * 3 + c];
                    byte p01 = rgbPixels[(y0 * srcWidth + x1) * 3 + c];
                    byte p10 = rgbPixels[(y1 * srcWidth + x0) * 3 + c];
                    byte p11 = rgbPixels[(y1 * srcWidth + x1) * 3 + c];
                    float v = (p00 * wx0 + p01 * wx1) * wy0 + (p10 * wx0 + p11 * wx1) * wy1;
                    outPtr[c * plane + dstPos] = (v / 255f - Mean[c]) / Std[c];
                }
            }
        }
        return output;
    }
}
