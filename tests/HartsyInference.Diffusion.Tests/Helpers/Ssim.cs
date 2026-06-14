namespace HartsyInference.Diffusion.Tests.Helpers;

/// <summary>
/// Structural Similarity Index (SSIM) for comparing two RGB images.
/// Uses an 11×11 Gaussian window with σ=1.5 (the standard configuration from Wang et al. 2004).
/// Operates per-channel (R, G, B) and returns the mean across channels.
///
/// Used by the cross-backend SSIM acceptance gates (Phase 3.5 §5):
///   • SD1.5 512×512 Vulkan vs CUDA — SSIM > 0.99
///   • SDXL 1024×1024 Vulkan vs CUDA — SSIM > 0.95
/// </summary>
public static class Ssim
{
    private const float K1 = 0.01f;
    private const float K2 = 0.03f;
    private const float DataRange = 255f;
    private const float C1 = (K1 * DataRange) * (K1 * DataRange);
    private const float C2 = (K2 * DataRange) * (K2 * DataRange);
    private const int WindowRadius = 5;     // 11×11 window
    private const int WindowSize = 2 * WindowRadius + 1;
    private const float Sigma = 1.5f;

    /// <summary>Computes mean SSIM between two interleaved RGB byte buffers (layout: <c>[H, W, 3]</c>).</summary>
    public static double Compute(byte[] rgbA, byte[] rgbB, int width, int height)
    {
        if (rgbA.Length != width * height * 3 || rgbB.Length != width * height * 3)
            throw new ArgumentException($"RGB buffer length mismatch: width={width}, height={height}, gotA={rgbA.Length}, gotB={rgbB.Length}");

        float[] window = BuildGaussianWindow();

        double rSsim = ChannelSsim(rgbA, rgbB, width, height, channelOffset: 0, window);
        double gSsim = ChannelSsim(rgbA, rgbB, width, height, channelOffset: 1, window);
        double bSsim = ChannelSsim(rgbA, rgbB, width, height, channelOffset: 2, window);
        return (rSsim + gSsim + bSsim) / 3.0;
    }

    private static float[] BuildGaussianWindow()
    {
        float[] w = new float[WindowSize * WindowSize];
        double twoSigmaSq = 2.0 * Sigma * Sigma;
        double sum = 0;
        for (int dy = -WindowRadius; dy <= WindowRadius; dy++)
            for (int dx = -WindowRadius; dx <= WindowRadius; dx++)
            {
                double v = Math.Exp(-(dx * dx + dy * dy) / twoSigmaSq);
                w[(dy + WindowRadius) * WindowSize + (dx + WindowRadius)] = (float)v;
                sum += v;
            }
        for (int i = 0; i < w.Length; i++) w[i] = (float)(w[i] / sum);
        return w;
    }

    private static double ChannelSsim(byte[] rgbA, byte[] rgbB, int width, int height, int channelOffset, float[] window)
    {
        double ssimSum = 0;
        long count = 0;

        for (int y = WindowRadius; y < height - WindowRadius; y++)
            for (int x = WindowRadius; x < width - WindowRadius; x++)
            {
                float muA = 0, muB = 0;
                for (int dy = -WindowRadius; dy <= WindowRadius; dy++)
                    for (int dx = -WindowRadius; dx <= WindowRadius; dx++)
                    {
                        float w = window[(dy + WindowRadius) * WindowSize + (dx + WindowRadius)];
                        int idxA = ((y + dy) * width + (x + dx)) * 3 + channelOffset;
                        muA += w * rgbA[idxA];
                        muB += w * rgbB[idxA];
                    }

                float varA = 0, varB = 0, covAB = 0;
                for (int dy = -WindowRadius; dy <= WindowRadius; dy++)
                    for (int dx = -WindowRadius; dx <= WindowRadius; dx++)
                    {
                        float w = window[(dy + WindowRadius) * WindowSize + (dx + WindowRadius)];
                        int idx = ((y + dy) * width + (x + dx)) * 3 + channelOffset;
                        float a = rgbA[idx] - muA;
                        float b = rgbB[idx] - muB;
                        varA += w * a * a;
                        varB += w * b * b;
                        covAB += w * a * b;
                    }

                float numerator = (2 * muA * muB + C1) * (2 * covAB + C2);
                float denominator = (muA * muA + muB * muB + C1) * (varA + varB + C2);
                ssimSum += numerator / denominator;
                count++;
            }

        return count == 0 ? 1.0 : ssimSum / count;
    }
}
