using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Resizes an RGB image to the SAM 3 input grid and maps <c>[0, 1]</c> to <c>[-1, 1]</c>. The upstream
/// processor stretches without letterboxing; bilinear here is not the antialiased resampler upstream uses.</summary>
internal static unsafe class Sam3Preprocessor
{
    /// <summary>Returns a new F32 <c>[1, 3, size, size]</c> tensor. The input stays owned by the caller.</summary>
    public static Tensor Prepare(IBackend backend, Tensor image, int size)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(image);
        if (image.DType != DType.F32)
        {
            throw new ArgumentException($"SAM 3 image must be F32, found {image.DType}.", nameof(image));
        }

        if (image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.Shape[2] <= 0 || image.Shape[3] <= 0)
        {
            throw new ArgumentException($"SAM 3 image must be [1, 3, H, W], found {image.Shape}.", nameof(image));
        }

        Tensor resized = new(new TensorShape(1, 3, size, size), DType.F32);
        backend.InterpolateBilinear2D(resized, image, alignCorners: false);

        float* data = (float*)resized.DataPointer;
        long count = 3L * size * size;
        for (long i = 0; i < count; i++)
        {
            data[i] = data[i] * 2f - 1f;
        }

        return resized;
    }
}
