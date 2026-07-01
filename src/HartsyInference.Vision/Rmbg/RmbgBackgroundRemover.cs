using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Rmbg;

/// <summary>End-to-end background removal with <see cref="BriaRmbg"/>: raw RGB → foreground alpha mask, and a
/// helper that composites the subject onto a neutral gray-0.5 background (what the image→3D pipelines — TripoSR /
/// Hunyuan3D — expect). This is the pure-C# replacement for the Python <c>rembg</c> + composite step, so raw
/// photos work without an external tool.</summary>
public sealed unsafe class RmbgBackgroundRemover
{
    private const int InputSize = 1024;
    private readonly BriaRmbg _net;

    public RmbgBackgroundRemover(BriaRmbg net) => _net = net;

    /// <summary>Foreground alpha for an interleaved-RGB24 image, at the source resolution, min-max normalized to
    /// [0,1] (matching RMBG's <c>postprocess_image</c>). Returns a <c>[srcHeight·srcWidth]</c> alpha buffer.</summary>
    public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int srcWidth, int srcHeight)
    {
        Tensor pixels = Preprocess(rgb, srcWidth, srcHeight);
        Tensor mask = _net.Forward(backend, pixels);          // [1,1,1024,1024] in [0,1]
        Tensor maskF = mask.DType == DType.F32 ? mask : mask.CastTo(DType.F32);
        float[] alpha = ResizeMinMax((float*)maskF.DataPointer, InputSize, InputSize, srcWidth, srcHeight);
        if (!ReferenceEquals(maskF, mask)) maskF.Dispose();
        mask.Dispose(); pixels.Dispose();
        return alpha;
    }

    /// <summary>Removes the background and composites the subject onto gray 0.5: <c>out = rgb·α + (1−α)·127.5</c>.
    /// Returns interleaved-RGB24 bytes at the source resolution — ready for the image→3D preprocessors.</summary>
    public byte[] CompositeOnGray(IBackend backend, ReadOnlySpan<byte> rgb, int srcWidth, int srcHeight)
    {
        float[] alpha = Alpha(backend, rgb, srcWidth, srcHeight);
        byte[] outRgb = new byte[(long)srcWidth * srcHeight * 3];
        for (long p = 0; p < (long)srcWidth * srcHeight; p++)
        {
            float a = alpha[p];
            for (int c = 0; c < 3; c++)
            {
                float v = rgb[(int)(p * 3 + c)] * a + (1f - a) * 127.5f;
                outRgb[p * 3 + c] = (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
            }
        }
        return outRgb;
    }

    /// <summary>Resize (bilinear) to 1024², quantize to uint8, then <c>(x/255 − 0.5)</c> → <c>[1,3,1024,1024]</c>,
    /// matching RMBG's <c>preprocess_image</c>.</summary>
    private static Tensor Preprocess(ReadOnlySpan<byte> rgb, int sw, int sh)
    {
        Tensor o = new(new TensorShape(1, 3, InputSize, InputSize), DType.F32);
        float* op = (float*)o.DataPointer;
        long plane = (long)InputSize * InputSize;
        float scaleY = (float)sh / InputSize, scaleX = (float)sw / InputSize;
        for (int oy = 0; oy < InputSize; oy++)
        {
            float syf = (oy + 0.5f) * scaleY - 0.5f;
            int y0 = (int)MathF.Floor(syf); float fy = syf - y0;
            int y0c = Math.Clamp(y0, 0, sh - 1), y1c = Math.Clamp(y0 + 1, 0, sh - 1);
            for (int ox = 0; ox < InputSize; ox++)
            {
                float sxf = (ox + 0.5f) * scaleX - 0.5f;
                int x0 = (int)MathF.Floor(sxf); float fx = sxf - x0;
                int x0c = Math.Clamp(x0, 0, sw - 1), x1c = Math.Clamp(x0 + 1, 0, sw - 1);
                long dst = (long)oy * InputSize + ox;
                for (int c = 0; c < 3; c++)
                {
                    float a = rgb[(y0c * sw + x0c) * 3 + c], b = rgb[(y0c * sw + x1c) * 3 + c];
                    float cc = rgb[(y1c * sw + x0c) * 3 + c], d = rgb[(y1c * sw + x1c) * 3 + c];
                    float v = (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy;
                    op[c * plane + dst] = ((byte)v) / 255f - 0.5f;   // uint8 quantize like utilities.preprocess_image
                }
            }
        }
        return o;
    }

    /// <summary>Resizes the [0,1] mask (bilinear) to the source size, then min-max normalizes to [0,1]
    /// (RMBG <c>postprocess_image</c>).</summary>
    private static float[] ResizeMinMax(float* mask, int mh, int mw, int dw, int dh)
    {
        float[] o = new float[(long)dw * dh];
        float scaleY = (float)mh / dh, scaleX = (float)mw / dw;
        float mn = float.MaxValue, mx = float.MinValue;
        for (int oy = 0; oy < dh; oy++)
        {
            float syf = (oy + 0.5f) * scaleY - 0.5f;
            int y0 = (int)MathF.Floor(syf); float fy = syf - y0;
            int y0c = Math.Clamp(y0, 0, mh - 1), y1c = Math.Clamp(y0 + 1, 0, mh - 1);
            for (int ox = 0; ox < dw; ox++)
            {
                float sxf = (ox + 0.5f) * scaleX - 0.5f;
                int x0 = (int)MathF.Floor(sxf); float fx = sxf - x0;
                int x0c = Math.Clamp(x0, 0, mw - 1), x1c = Math.Clamp(x0 + 1, 0, mw - 1);
                float a = mask[y0c * mw + x0c], b = mask[y0c * mw + x1c], cc = mask[y1c * mw + x0c], d = mask[y1c * mw + x1c];
                float v = (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy;
                o[(long)oy * dw + ox] = v; mn = Math.Min(mn, v); mx = Math.Max(mx, v);
            }
        }
        float range = mx - mn < 1e-8f ? 1f : mx - mn;
        for (long i = 0; i < o.Length; i++) o[i] = (o[i] - mn) / range;
        return o;
    }
}
