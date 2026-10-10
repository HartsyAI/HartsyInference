using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Clip;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Pure-math tests for <see cref="ClipImagePreprocessor"/>. No model weights required.</summary>
public sealed class ClipImagePreprocessorTests
{
    [Fact]
    public void Preprocess_ConstantGrayImage_AllPixelsHaveExpectedNormalizedValue()
    {
        ClipImagePreprocessor pre = new();
        byte[] rgb = new byte[224 * 224 * 3];
        Array.Fill<byte>(rgb, 128);

        using Tensor output = pre.Preprocess(rgb, 224, 224);
        ReadOnlySpan<float> data = output.AsReadOnlySpan<float>();

        // 128/255 = 0.5019607...; normalize: (0.5019607 - mean) / std per channel.
        float[] mean = ClipImagePreprocessor.DefaultMean;
        float[] std = ClipImagePreprocessor.DefaultStd;
        float gray = 128f / 255f;
        float[] expected = [
            (gray - mean[0]) / std[0],
            (gray - mean[1]) / std[1],
            (gray - mean[2]) / std[2],
        ];

        int planeSize = 224 * 224;
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < planeSize; i++)
            {
                Assert.InRange(data[c * planeSize + i], expected[c] - 1e-4f, expected[c] + 1e-4f);
            }
        }
    }

    [Fact]
    public void Preprocess_RejectsMismatchedBufferLength()
    {
        ClipImagePreprocessor pre = new();
        byte[] tooSmall = new byte[10];
        Assert.Throws<ArgumentException>(() => pre.Preprocess(tooSmall, 100, 100));
    }

    [Fact]
    public void Preprocess_TallImage_CenterCropsToSquare()
    {
        ClipImagePreprocessor pre = new();
        // 100 wide, 300 tall. Shortest edge is 100 → scale = 224/100 = 2.24. Resized: 224 × 672.
        // Center crop to 224 × 224 takes rows 224..448 of the resized image.
        // Make the top stripe red, bottom stripe blue, middle stripe green.
        byte[] rgb = new byte[100 * 300 * 3];
        for (int y = 0; y < 300; y++)
        {
            for (int x = 0; x < 100; x++)
            {
                int idx = (y * 100 + x) * 3;
                if (y < 100) { rgb[idx + 0] = 255; }      // red top
                else if (y < 200) { rgb[idx + 1] = 255; } // green middle
                else { rgb[idx + 2] = 255; }              // blue bottom
            }
        }

        using Tensor output = pre.Preprocess(rgb, 100, 300);
        ReadOnlySpan<float> data = output.AsReadOnlySpan<float>();

        // After center crop, the middle of the output should be dominated by the green channel.
        // Sample the center pixel (112, 112) — should have the largest value in the G channel.
        int planeSize = 224 * 224;
        int centerIdx = 112 * 224 + 112;
        float r = data[0 * planeSize + centerIdx];
        float g = data[1 * planeSize + centerIdx];
        float b = data[2 * planeSize + centerIdx];

        // The G channel's mean is 0.4578 with std 0.2613 — the normalized value for pure G=1.0 is
        // (1 - 0.4578) / 0.2613 ~= 2.07; R and B at value 0 are ~ -1.79 / -1.48. So G > R, G > B.
        Assert.True(g > r, $"Expected center pixel to be dominated by green: r={r} g={g} b={b}");
        Assert.True(g > b, $"Expected center pixel to be dominated by green: r={r} g={g} b={b}");
    }
}
