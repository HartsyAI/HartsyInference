using HartsyInference.Core.Imaging;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Resizes an RGB image to the SAM 3 input grid and maps <c>[0, 1]</c> to <c>[-1, 1]</c>. The upstream processor
/// takes 8-bit pixels, resizes them with antialiased bilinear (8-bit intermediate, as PIL and torchvision do), then normalises.</summary>
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

        int height = (int)image.Shape[2], width = (int)image.Shape[3];
        int plane = width * height;
        float* source = (float*)image.DataPointer;
        byte[] interleaved = new byte[(long)plane * 3];
        for (int i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                interleaved[i * 3 + c] = (byte)Math.Clamp(MathF.Round(source[c * plane + i] * 255f), 0f, 255f);
            }
        }

        float[] resized = new float[(long)size * size * 3];
        Resample.BilinearHwc8(interleaved, width, height, 3, resized, size, size, antialias: true, quantizeU8: true);

        Tensor output = new(new TensorShape(1, 3, size, size), DType.F32);
        float* data = (float*)output.DataPointer;
        int outPlane = size * size;
        for (int i = 0; i < outPlane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[c * outPlane + i] = resized[i * 3 + c] / 255f * 2f - 1f;
            }
        }

        return output;
    }
}
