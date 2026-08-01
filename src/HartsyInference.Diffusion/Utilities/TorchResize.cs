namespace HartsyInference.Diffusion.Utilities;

/// <summary>Bicubic resize matching PyTorch's antialiased tensor path (<c>F.interpolate(..., mode="bicubic",
/// antialias=True)</c>, ATen <c>_upsample_bicubic2d_aa</c>) — the path torchvision's <c>TVF.resize</c> takes on
/// tensors since 0.17. Cubic kernel a=−0.5 (the AA path mimics PIL's Catmull-Rom — a=−0.75 is torch's
/// NON-antialiased kernel, empirically ruled out at 4e-2 maxAbs); on downscale the support widens; at borders
/// taps are truncated and weights renormalized (NOT edge-clamped — that is the non-AA kernel, which torchvision
/// no longer uses by default). Separable: width pass then height pass, per-output tap weights precomputed once
/// per axis, entirely in float32 to mirror ATen's scalar_t arithmetic (validated ≤1e-5 maxAbs in
/// SeedVr2PreprocessParityTests).</summary>
public static class TorchResize
{
    private const float A = -0.5f;

    /// <summary>Resizes one CHW fp32 plane stack. Values are passed through unclamped (bicubic overshoots,
    /// PyTorch keeps the overshoot — clamp at the call site only if the reference chain does).</summary>
    /// <param name="src">Source, C×srcH×srcW, row-major planes.</param>
    /// <param name="dst">Destination, C×dstH×dstW; must be sized by the caller.</param>
    public static void BicubicAntialiasChw(
        ReadOnlySpan<float> src, int channels, int srcH, int srcW,
        Span<float> dst, int dstH, int dstW)
    {
        if (src.Length < (long)channels * srcH * srcW)
            throw new ArgumentException($"Source too small: {src.Length} < {channels}x{srcH}x{srcW}.");
        if (dst.Length < (long)channels * dstH * dstW)
            throw new ArgumentException($"Destination too small: {dst.Length} < {channels}x{dstH}x{dstW}.");

        AxisWeights wx = ComputeAxisWeights(srcW, dstW);
        AxisWeights wy = ComputeAxisWeights(srcH, dstH);

        // Width pass: (srcH, srcW) -> (srcH, dstW), then height pass -> (dstH, dstW). ATen's separable
        // kernel resolves the last (width) dimension first; matching the order keeps float drift minimal.
        float[] mid = new float[srcH * dstW];
        for (int c = 0; c < channels; c++)
        {
            ReadOnlySpan<float> plane = src.Slice(c * srcH * srcW, srcH * srcW);
            for (int y = 0; y < srcH; y++)
            {
                ReadOnlySpan<float> row = plane.Slice(y * srcW, srcW);
                for (int x = 0; x < dstW; x++)
                {
                    int first = wx.First[x];
                    int taps = wx.Count[x];
                    int wBase = x * wx.MaxTaps;
                    float acc = 0f;
                    for (int k = 0; k < taps; k++)
                        acc += row[first + k] * wx.Weights[wBase + k];
                    mid[y * dstW + x] = acc;
                }
            }

            Span<float> outPlane = dst.Slice(c * dstH * dstW, dstH * dstW);
            for (int y = 0; y < dstH; y++)
            {
                int first = wy.First[y];
                int taps = wy.Count[y];
                int wBase = y * wy.MaxTaps;
                for (int x = 0; x < dstW; x++)
                {
                    float acc = 0f;
                    for (int k = 0; k < taps; k++)
                        acc += mid[(first + k) * dstW + x] * wy.Weights[wBase + k];
                    outPlane[y * dstW + x] = acc;
                }
            }
        }
    }

    private readonly record struct AxisWeights(int[] First, int[] Count, float[] Weights, int MaxTaps);

    // ATen computes scale/center/filter in scalar_t (float32 for f32 tensors) — weight drift grows
    // linearly with output index if double is used here (empirically 3.4e-5 at i≈1000; float32 matches
    // the impulse response to ~1e-7). Keep ALL of this in float on purpose.
    private static AxisWeights ComputeAxisWeights(int inSize, int outSize)
    {
        float scale = inSize / (float)outSize;
        float support = scale >= 1f ? 2f * scale : 2f;
        float invscale = scale >= 1f ? 1f / scale : 1f;
        int maxTaps = (int)MathF.Ceiling(support) * 2 + 1;

        int[] first = new int[outSize];
        int[] count = new int[outSize];
        float[] weights = new float[outSize * maxTaps];
        Span<float> raw = stackalloc float[maxTaps];

        for (int i = 0; i < outSize; i++)
        {
            float center = scale * (i + 0.5f);
            int xmin = Math.Max((int)(center - support + 0.5f), 0);
            int xsize = Math.Min((int)(center + support + 0.5f), inSize) - xmin;

            float total = 0f;
            for (int j = 0; j < xsize; j++)
            {
                float w = CubicFilter((j + xmin - center + 0.5f) * invscale);
                raw[j] = w;
                total += w;
            }

            first[i] = xmin;
            count[i] = xsize;
            for (int j = 0; j < xsize; j++)
                weights[i * maxTaps + j] = total != 0f ? raw[j] / total : 0f;
        }
        return new AxisWeights(first, count, weights, maxTaps);
    }

    private static float CubicFilter(float x)
    {
        x = MathF.Abs(x);
        if (x < 1f)
            return ((A + 2f) * x - (A + 3f)) * x * x + 1f;
        if (x < 2f)
            return (((x - 5f) * x + 8f) * x - 4f) * A;
        return 0f;
    }
}
