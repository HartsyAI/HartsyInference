using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Codec;

/// <summary>Conversions between interleaved HWC rgb24 buffers and planar NCHW image tensors, shared by the
/// vision pipelines that hand images to/from the backend (upscale input, ControlNet preprocessor outputs).</summary>
public static unsafe class ImageTensor
{
    /// <summary>HWC rgb24 bytes [0,255] → NCHW F32 <c>[1,3,H,W]</c> in [0,1].</summary>
    public static Tensor RgbToTensor01(ReadOnlySpan<byte> rgb, int width, int height)
    {
        long expected = (long)width * height * 3;
        if (width <= 0 || height <= 0 || rgb.Length != expected)
            throw new ArgumentException($"rgb length {rgb.Length} != {width}x{height}x3.", nameof(rgb));

        Tensor t = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* o = (float*)t.DataPointer;
        int plane = width * height;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int px = (y * width + x) * 3;
                int dst = y * width + x;
                o[dst] = rgb[px] * (1f / 255f);
                o[plane + dst] = rgb[px + 1] * (1f / 255f);
                o[2 * plane + dst] = rgb[px + 2] * (1f / 255f);
            }
        }
        return t;
    }
}
