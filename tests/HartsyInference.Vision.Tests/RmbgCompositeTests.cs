using HartsyInference.Vision.Rmbg;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>The two consumers of one RMBG matte — the gray composite the 3-D preprocessors read and the 8-bit alpha
/// a PNG carries — must agree on the same alpha, or a cutout's edge pixels drift between them.</summary>
public sealed class RmbgCompositeTests
{
    [Fact]
    public void CompositeOnGray_BlendsTowardMidGrayByAlpha()
    {
        byte[] rgb = [255, 0, 0, 255, 0, 0, 255, 0, 0];
        float[] alpha = [1f, 0f, 0.5f];

        byte[] result = RmbgBackgroundRemover.CompositeOnGray(alpha, rgb, 3, 1);

        Assert.Equal(new byte[] { 255, 0, 0 }, result[..3]);            // opaque: source pixel
        Assert.Equal(new byte[] { 128, 128, 128 }, result[3..6]);       // transparent: gray 127.5 rounded
        Assert.Equal(new byte[] { 191, 64, 64 }, result[6..9]);         // half: midway
    }

    [Fact]
    public void AlphaToBytes_QuantizesAndClamps()
    {
        byte[] bytes = RmbgBackgroundRemover.AlphaToBytes([0f, 0.5f, 1f, 1.5f, -0.2f]);
        Assert.Equal(new byte[] { 0, 128, 255, 255, 0 }, bytes);
    }

    [Fact]
    public void CompositeOnGray_RejectsMismatchedAlpha()
    {
        Assert.Throws<ArgumentException>(() => RmbgBackgroundRemover.CompositeOnGray(new float[2], new byte[9], 3, 1));
    }
}
