using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>The processor's resize geometry (<c>navit_resize_image</c>) and the normalized tensor it hands the tower.</summary>
public sealed class KimiImagePreprocessorTests
{
    private static readonly KimiImageSpec Published = new()
    {
        PatchSize = 14,
        MergeKernelSize = 2,
        InPatchLimit = 65536,
        PatchLimitOnOneSide = 512,
        ImageMean = [0.5f, 0.5f, 0.5f],
        ImageStd = [0.5f, 0.5f, 0.5f],
    };

    [Theory]
    // Within the budget: no shrink, and the pad rounds each side up to a 28-pixel merge block.
    [InlineData(448, 448, 448, 448, 0, 0, 256)]
    [InlineData(1000, 500, 1000, 500, 8, 4, 648)]
    // Wider than the side limit of 512 patches (7168 pixels): shrunk to it, exact in binary.
    [InlineData(8192, 64, 7168, 56, 0, 0, 512)]
    public void Resize_shrinks_only_past_the_limits_and_pads_to_whole_merge_blocks(int width, int height, int newWidth, int newHeight,
        int padWidth, int padHeight, int tokens)
    {
        KimiImageResize resize = KimiImagePreprocessor.Resize(width, height, Published);

        Assert.Equal(new KimiImageResize(newWidth, newHeight, padWidth, padHeight, tokens), resize);
    }

    [Fact]
    public void Preprocess_resizes_normalizes_and_pads_with_black_before_normalization()
    {
        byte[] rgb = new byte[4 * 2 * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i * 37 % 256);
        using Tensor pixels = KimiImagePreprocessor.Preprocess(rgb, width: 4, height: 2, Published);

        Assert.Equal(1L, pixels.Shape[0]);
        Assert.Equal(3L, pixels.Shape[1]);
        Assert.Equal(28L, pixels.Shape[2]);
        Assert.Equal(28L, pixels.Shape[3]);
        ReadOnlySpan<float> values = pixels.AsReadOnlySpan<float>();
        const int plane = 28 * 28;
        // Identity resize at the same size: the pixel at (x, y), channel c, normalizes from its own byte.
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    float expected = (rgb[(y * 4 + x) * 3 + c] / 255f - 0.5f) / 0.5f;
                    Assert.InRange(MathF.Abs(values[c * plane + y * 28 + x] - expected), 0f, 1e-5f);
                }
            }
        }
        // Outside the 4x2 image the pad is raw 0, which normalizes to -mean/std = -1.
        Assert.Equal(-1f, values[27 * 28 + 27]);
        Assert.Equal(-1f, values[5]);
        Assert.Equal(-1f, values[2 * 28]);
    }

    [Fact]
    public void Spec_refuses_fixed_output_tokens_by_name()
    {
        using JsonDocument doc = JsonDocument.Parse(
            "{\"patch_size\": 14, \"merge_kernel_size\": 2, \"in_patch_limit\": 65536, \"patch_limit_on_one_side\": 512, "
            + "\"fixed_output_tokens\": 256, \"image_mean\": [0.5, 0.5, 0.5], \"image_std\": [0.5, 0.5, 0.5]}");

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => KimiImageSpec.Read(doc.RootElement));

        Assert.Contains("fixed_output_tokens", refused.Message, StringComparison.Ordinal);
    }
}
