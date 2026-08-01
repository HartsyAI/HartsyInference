namespace HartsyInference.Video.Pipelines;

/// <summary>StableSR-style wavelet color/low-frequency transplant — the optional <c>color_fix.py</c> the
/// SeedVR2 reference loads IF PRESENT (absent by default; the pipeline default matches the reference's
/// no-transplant path). 5-level dilated 3×3 Gaussian pyramid: output keeps the model's high frequencies and
/// takes the low-frequency band from the (bicubic-resized) input, lerped by <c>1−strength</c>. Guards the
/// documented oversharpening failure mode on lightly-degraded input.</summary>
public static class SeedVr2ColorFix
{
    private const int Levels = 5;

    /// <summary>Blends per-plane: <c>model + (1−strength)·(inputLow − modelLow)</c>. Both buffers are
    /// C×H×W planes in [-1,1]; <paramref name="model"/> is modified in place.</summary>
    public static void Transplant(Span<float> model, ReadOnlySpan<float> input, int channels, int h, int w, float strength)
    {
        if (strength >= 1f)
            return;
        float mix = 1f - strength;
        float[] modelLow = new float[h * w];
        float[] inputLow = new float[h * w];
        float[] scratch = new float[h * w];
        for (int c = 0; c < channels; c++)
        {
            Span<float> plane = model.Slice(c * h * w, h * w);
            LowFrequency(plane, modelLow, scratch, h, w);
            LowFrequency(input.Slice(c * h * w, h * w), inputLow, scratch, h, w);
            for (int i = 0; i < plane.Length; i++)
                plane[i] += mix * (inputLow[i] - modelLow[i]);
        }
    }

    private static void LowFrequency(ReadOnlySpan<float> src, float[] dst, float[] scratch, int h, int w)
    {
        src.CopyTo(dst);
        for (int level = 0; level < Levels; level++)
        {
            WaveletBlur(dst, scratch, h, w, 1 << level);
            scratch.CopyTo(dst, 0);
        }
    }

    /// <summary>3×3 [[1,2,1],[2,4,2],[1,2,1]]/16 blur at the given dilation, replicate-padded.</summary>
    private static void WaveletBlur(ReadOnlySpan<float> src, float[] dst, int h, int w, int dilation)
    {
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float acc = 0;
            for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++)
            {
                int sy = Math.Clamp(y + ky * dilation, 0, h - 1);
                int sx = Math.Clamp(x + kx * dilation, 0, w - 1);
                float weight = (ky == 0 ? 2f : 1f) * (kx == 0 ? 2f : 1f) / 16f;
                acc += src[sy * w + sx] * weight;
            }
            dst[y * w + x] = acc;
        }
    }
}
