using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR image preprocessing: resize the source RGB image to a fixed square
/// (<see cref="TargetSize"/>×<see cref="TargetSize"/>, default 640) with a PIL-compatible bilinear
/// (triangle) filter — antialiased when downscaling, matching <c>RTDetrImageProcessor</c> — then
/// rescale to [0,1] (<c>/255</c>). RT-DETR does <b>not</b> letterbox, pad, or mean/std-normalize; boxes
/// come back normalized to the resized square so they map to the source by a plain axis scale.</summary>
public sealed class RtDetrPreprocessor
{
    /// <summary>Square target edge fed to the backbone (640).</summary>
    public int TargetSize { get; }

    /// <summary>Creates a preprocessor for the given square target size.</summary>
    public RtDetrPreprocessor(int targetSize = 640)
    {
        if (targetSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize, "Target size must be positive.");
        TargetSize = targetSize;
    }

    /// <summary>Resizes and rescales <paramref name="rgbPixels"/> (HWC-packed RGB u8) to a
    /// <c>[1, 3, TargetSize, TargetSize]</c> F32 tensor in [0,1]. Owns the returned tensor.</summary>
    public Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        if (rgbPixels.Length < (long)width * height * 3)
            throw new ArgumentException("RGB buffer too small for the given width/height.", nameof(rgbPixels));
        int dst = TargetSize;
        float[] resized = ResizeBilinearPil(rgbPixels, width, height, dst, dst);   // [dst*dst*3], 0..255

        Tensor tensor = new Tensor(new TensorShape(1, 3, dst, dst), DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        int plane = dst * dst;
        for (int y = 0; y < dst; y++)
            for (int x = 0; x < dst; x++)
            {
                int src = (y * dst + x) * 3;
                int pix = y * dst + x;
                span[0 * plane + pix] = resized[src + 0] / 255f;
                span[1 * plane + pix] = resized[src + 1] / 255f;
                span[2 * plane + pix] = resized[src + 2] / 255f;
            }
        return tensor;
    }

    /// <summary>Separable PIL-style bilinear resize (triangle filter, support scaled by the reduction
    /// factor for antialiasing) of an HWC RGB u8 image. Returns HWC float in [0,255], rounded to
    /// integers after each pass to match PIL's uint8 intermediate.</summary>
    internal static float[] ResizeBilinearPil(ReadOnlySpan<byte> src, int srcW, int srcH, int dstW, int dstH)
    {
        // Horizontal pass: srcW -> dstW, height stays srcH.
        (int[] bounds, float[] weights, int ksize) = BuildCoefficients(srcW, dstW);
        float[] horiz = new float[(long)srcH * dstW * 3];
        for (int y = 0; y < srcH; y++)
            for (int ox = 0; ox < dstW; ox++)
            {
                int xmin = bounds[ox * 2];
                int xn = bounds[ox * 2 + 1];
                float r = 0f, g = 0f, b = 0f;
                for (int k = 0; k < xn; k++)
                {
                    float wgt = weights[ox * ksize + k];
                    int sx = xmin + k;
                    int si = (y * srcW + sx) * 3;
                    r += src[si] * wgt;
                    g += src[si + 1] * wgt;
                    b += src[si + 2] * wgt;
                }
                int di = (y * dstW + ox) * 3;
                horiz[di] = RoundClip(r);
                horiz[di + 1] = RoundClip(g);
                horiz[di + 2] = RoundClip(b);
            }

        // Vertical pass: srcH -> dstH, width stays dstW.
        (int[] vbounds, float[] vweights, int vksize) = BuildCoefficients(srcH, dstH);
        float[] outp = new float[(long)dstH * dstW * 3];
        for (int oy = 0; oy < dstH; oy++)
        {
            int ymin = vbounds[oy * 2];
            int yn = vbounds[oy * 2 + 1];
            for (int x = 0; x < dstW; x++)
            {
                float r = 0f, g = 0f, b = 0f;
                for (int k = 0; k < yn; k++)
                {
                    float wgt = vweights[oy * vksize + k];
                    int sy = ymin + k;
                    int si = (sy * dstW + x) * 3;
                    r += horiz[si] * wgt;
                    g += horiz[si + 1] * wgt;
                    b += horiz[si + 2] * wgt;
                }
                int di = (oy * dstW + x) * 3;
                outp[di] = RoundClip(r);
                outp[di + 1] = RoundClip(g);
                outp[di + 2] = RoundClip(b);
            }
        }
        return outp;
    }

    /// <summary>Builds PIL's per-output-pixel triangle-filter coefficients for a 1D resize
    /// <paramref name="inSize"/> → <paramref name="outSize"/>. Returns (bounds[out·2] = {start, count},
    /// weights[out·ksize], ksize).</summary>
    private static (int[] Bounds, float[] Weights, int KSize) BuildCoefficients(int inSize, int outSize)
    {
        const float support = 1.0f;   // bilinear
        float scale = (float)inSize / outSize;
        float filterScale = scale < 1.0f ? 1.0f : scale;
        float actualSupport = support * filterScale;
        int ksize = (int)MathF.Ceiling(actualSupport) * 2 + 1;
        int[] bounds = new int[outSize * 2];
        float[] weights = new float[(long)outSize * ksize];
        for (int ox = 0; ox < outSize; ox++)
        {
            float center = (ox + 0.5f) * scale;
            int xmin = (int)(center - actualSupport + 0.5f);
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + actualSupport + 0.5f);
            if (xmax > inSize) xmax = inSize;
            int n = xmax - xmin;
            float ss = 1.0f / filterScale;
            float total = 0f;
            for (int k = 0; k < n; k++)
            {
                float t = (xmin + k + 0.5f - center) * ss;
                float w = 1.0f - MathF.Abs(t);   // triangle
                if (w < 0f) w = 0f;
                weights[ox * ksize + k] = w;
                total += w;
            }
            if (total > 0f)
                for (int k = 0; k < n; k++)
                    weights[ox * ksize + k] /= total;
            bounds[ox * 2] = xmin;
            bounds[ox * 2 + 1] = n;
        }
        return (bounds, weights, ksize);
    }

    private static float RoundClip(float v)
    {
        int r = (int)MathF.Round(v, MidpointRounding.ToEven);
        if (r < 0) r = 0; else if (r > 255) r = 255;
        return r;
    }
}
