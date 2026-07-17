using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Siglip;

/// <summary>SigLIP image preprocessing — square resize + symmetric normalize. Differs
/// from CLIP in two ways:
/// <list type="bullet">
///   <item><b>No aspect-preserving short-edge resize + center crop.</b> SigLIP stretches directly to <c>imageSize × imageSize</c>.</item>
///   <item><b>Symmetric normalize</b>: mean=std=[0.5, 0.5, 0.5] (the Inception convention) rather than CLIP's per-channel ImageNet-derived values.</item>
/// </list>
/// The resize is antialiased bicubic (a = −0.5), matching HF <c>SiglipImageProcessor</c>'s PIL
/// <c>BICUBIC</c> path. A plain (non-antialiased) kernel aliases on downscales and measurably shifts the
/// patch-token embeddings (FLUX.1 Redux conditioning corr dropped to 0.94 on a 810×1080 → 384² input).
/// Input is HWC-packed RGB u8; output is <c>[1, 3, size, size]</c> F32 ready for the SigLIP
/// vision encoder.</summary>
public sealed unsafe class SiglipImagePreprocessor
{
    private readonly int _imageSize;

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

        float[] resized = new float[(long)_imageSize * _imageSize * 3];
        Resample.BicubicHwc8(rgbPixels, srcWidth, srcHeight, 3, resized, _imageSize, _imageSize,
            a: -0.5f, antialias: true, quantizeU8: true);

        Tensor output = new(new TensorShape(1, 3, _imageSize, _imageSize), DType.F32);
        float* outPtr = (float*)output.DataPointer;
        long plane = (long)_imageSize * _imageSize;
        for (long i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                // /255 → normalize: (v/255 - 0.5) / 0.5 = v/127.5 - 1
                outPtr[c * plane + i] = resized[i * 3 + c] * (1f / 127.5f) - 1f;
            }
        }
        return output;
    }
}
