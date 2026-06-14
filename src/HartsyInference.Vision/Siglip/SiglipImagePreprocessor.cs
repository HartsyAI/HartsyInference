using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Siglip;

/// <summary>SigLIP image preprocessing — square bilinear resize + symmetric normalize. Differs
/// from CLIP in two ways:
/// <list type="bullet">
///   <item><b>No aspect-preserving short-edge resize + center crop.</b> SigLIP stretches directly to <c>imageSize × imageSize</c>.</item>
///   <item><b>Symmetric normalize</b>: mean=std=[0.5, 0.5, 0.5] (the Inception convention) rather than CLIP's per-channel ImageNet-derived values.</item>
/// </list>
/// Input is HWC-packed RGB u8; output is <c>[1, 3, size, size]</c> F32 ready for the SigLIP
/// vision encoder.</summary>
public sealed unsafe class SiglipImagePreprocessor
{
    private readonly int _imageSize;
    private const float Mean = 0.5f;
    private const float InvStd = 1f / 0.5f; // = 2.0

    public int ImageSize => _imageSize;

    public SiglipImagePreprocessor(int imageSize)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "Image size must be positive.");
        _imageSize = imageSize;
    }

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

        // Bilinear resize directly to imageSize × imageSize. Apply normalize inline.
        float scaleX = (float)srcWidth / _imageSize;
        float scaleY = (float)srcHeight / _imageSize;

        for (int oy = 0; oy < _imageSize; oy++)
        {
            float sy = (oy + 0.5f) * scaleY - 0.5f;
            int sy0 = (int)MathF.Floor(sy);
            float fy = sy - sy0;
            int y0 = Math.Clamp(sy0, 0, srcHeight - 1);
            int y1 = Math.Clamp(sy0 + 1, 0, srcHeight - 1);
            float wy1 = fy, wy0 = 1f - fy;

            for (int ox = 0; ox < _imageSize; ox++)
            {
                float sx = (ox + 0.5f) * scaleX - 0.5f;
                int sx0 = (int)MathF.Floor(sx);
                float fx = sx - sx0;
                int x0 = Math.Clamp(sx0, 0, srcWidth - 1);
                int x1 = Math.Clamp(sx0 + 1, 0, srcWidth - 1);
                float wx1 = fx, wx0 = 1f - fx;

                long dstPos = oy * _imageSize + ox;
                for (int c = 0; c < 3; c++)
                {
                    byte p00 = rgbPixels[(y0 * srcWidth + x0) * 3 + c];
                    byte p01 = rgbPixels[(y0 * srcWidth + x1) * 3 + c];
                    byte p10 = rgbPixels[(y1 * srcWidth + x0) * 3 + c];
                    byte p11 = rgbPixels[(y1 * srcWidth + x1) * 3 + c];
                    float v = (p00 * wx0 + p01 * wx1) * wy0 + (p10 * wx0 + p11 * wx1) * wy1;
                    // /255 → normalize: (v/255 - 0.5) / 0.5 = v/127.5 - 1
                    outPtr[c * plane + dstPos] = v * (1f / 127.5f) - 1f;
                }
            }
        }
        return output;
    }
}
