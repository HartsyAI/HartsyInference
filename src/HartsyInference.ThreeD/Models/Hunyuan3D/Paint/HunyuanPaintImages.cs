using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Image plumbing of the Paint pipeline: PIL-style bicubic resizing, and moving views between byte images, tensors and host arrays.</summary>
internal static unsafe class HunyuanPaintImages
{
    /// <summary>PIL <c>BICUBIC</c> (a = -0.5, antialiased, rounded to bytes).</summary>
    internal static byte[] Resize(ReadOnlySpan<byte> rgb, int width, int height, int targetWidth, int targetHeight)
    {
        if (width == targetWidth && height == targetHeight)
            return rgb.ToArray();
        float[] resized = new float[targetWidth * targetHeight * 3];
        Resample.BicubicHwc8(rgb, width, height, 3, resized, targetWidth, targetHeight, -0.5f, antialias: true, quantizeU8: true);
        byte[] bytes = new byte[resized.Length];
        for (int i = 0; i < bytes.Length; i++)
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(resized[i]), 0, 255);
        return bytes;
    }

    /// <summary>Stacks RGB byte images (HWC, square) into <c>[N, 3, S, S]</c> floats in <c>[-1, 1]</c>.</summary>
    internal static Tensor ToTensor(IReadOnlyList<byte[]> images, int size)
    {
        Tensor tensor = new(new TensorShape(images.Count, 3, size, size), DType.F32);
        float* p = (float*)tensor.DataPointer;
        long plane = (long)size * size;
        for (int v = 0; v < images.Count; v++)
            for (int i = 0; i < plane; i++)
                for (int c = 0; c < 3; c++)
                    p[(v * 3L + c) * plane + i] = images[v][i * 3 + c] / 255f * 2f - 1f;
        return tensor;
    }

    /// <summary>Converts a <c>[N, 3, H, W]</c> image tensor in about <c>[-1, 1]</c> to RGB bytes per image: <c>(x / 2 + 0.5)</c> clamped and rounded.</summary>
    internal static byte[][] ToBytes(Tensor images)
    {
        int n = (int)images.Shape[0], h = (int)images.Shape[2], w = (int)images.Shape[3];
        byte[][] result = new byte[n][];
        float* p = (float*)images.DataPointer;
        long plane = (long)h * w;
        for (int v = 0; v < n; v++)
        {
            result[v] = new byte[plane * 3];
            for (long i = 0; i < plane; i++)
                for (int c = 0; c < 3; c++)
                    result[v][i * 3 + c] = (byte)Math.Clamp((int)MathF.Round((p[(v * 3L + c) * plane + i] / 2f + 0.5f) * 255f), 0, 255);
        }
        return result;
    }

    /// <summary>Copies a tensor's floats to a host array.</summary>
    internal static float[] ToArray(Tensor tensor)
    {
        float[] data = new float[tensor.ElementCount];
        new ReadOnlySpan<float>((float*)tensor.DataPointer, data.Length).CopyTo(data);
        return data;
    }

    /// <summary>Creates an F32 tensor holding <paramref name="data"/>.</summary>
    internal static Tensor FromArray(float[] data, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((float*)tensor.DataPointer, data.Length));
        return tensor;
    }
}
