using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Annotators;

/// <summary>NormalBAE pre/post-processing matching comfyui_controlnet_aux's <c>NormalBaeDetector</c>: RGB
/// /255 + ImageNet normalization in, xyz normal out, mapped to the RGB conditioning image via
/// <c>(normal + 1)/2</c> and the Python pipeline's uint8 truncation. Host code by design: runs once per
/// generation, outside the per-step hot path. The caller resizes the source image; dimensions must be
/// multiples of 32 (controlnet_aux snaps to multiples of 64).</summary>
public sealed class NormalBaePreprocessor
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly NormalBaeModel _model;

    public NormalBaePreprocessor(NormalBaeModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    /// <summary>Wrapped model — exposed for weight preloading.</summary>
    public NormalBaeModel Model => _model;

    /// <summary>Converts interleaved RGB24 to the ImageNet-normalized network input <c>[1,3,H,W]</c>.</summary>
    public static unsafe Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        long expected = (long)width * height * 3;
        if (width <= 0 || height <= 0 || rgbPixels.Length != expected)
            throw new ArgumentException($"rgb length {rgbPixels.Length} != {width}x{height}x3.", nameof(rgbPixels));

        Tensor t = new(new TensorShape(1, 3, height, width), DType.F32);
        float* o = (float*)t.DataPointer;
        long plane = (long)width * height;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                long px = ((long)y * width + x) * 3;
                long dst = (long)y * width + x;
                o[dst] = (rgbPixels[(int)px] / 255f - Mean[0]) / Std[0];
                o[plane + dst] = (rgbPixels[(int)px + 1] / 255f - Mean[1]) / Std[1];
                o[2 * plane + dst] = (rgbPixels[(int)px + 2] / 255f - Mean[2]) / Std[2];
            }
        }
        return t;
    }

    /// <summary>Runs the detector and returns the conditioning image as interleaved RGB24
    /// (xyz normal mapped to <c>(n+1)/2·255</c> with numpy truncation).</summary>
    public byte[] Process(IBackend backend, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        using Tensor input = Preprocess(rgbPixels, width, height);
        Tensor normals = _model.Forward(backend, input);
        byte[] rgb = PostprocessToRgb24(normals);
        normals.Dispose();
        return rgb;
    }

    /// <summary>Model output <c>[1,4,H,W]</c> → interleaved RGB24 conditioning image, replicating the
    /// Python pipeline exactly: <c>((normal+1)·0.5).clip(0,1)</c> then <c>(·255).clip.astype(uint8)</c>
    /// (truncation) per xyz channel.</summary>
    public static unsafe byte[] PostprocessToRgb24(Tensor normals)
    {
        if (normals.Shape.Rank != 4 || normals.Shape[0] != 1 || normals.Shape[1] < 3)
            throw new ArgumentException($"normals must be [1,>=3,H,W]; got {normals.Shape}.", nameof(normals));
        int h = (int)normals.Shape[2], w = (int)normals.Shape[3];
        long plane = (long)h * w;
        float* src = (float*)normals.DataPointer;
        byte[] rgb = new byte[plane * 3];
        for (long i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float unit = Math.Clamp((src[c * plane + i] + 1f) * 0.5f, 0f, 1f);
                rgb[i * 3 + c] = HedPreprocessor.QuantizeU8(unit);
            }
        }
        return rgb;
    }
}
