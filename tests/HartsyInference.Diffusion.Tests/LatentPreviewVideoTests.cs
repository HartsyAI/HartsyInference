using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Latent2rgb preview coverage for video architectures and full temporal preview decoding.
/// Factors are Comfy's published tables — numeric hue accuracy is
/// inherently approximate; these tests verify wiring, shapes, and finite output.</summary>
public unsafe class LatentPreviewVideoTests
{
    /// <summary>Every architecture with a registered static latent preview and its canonical channel count.</summary>
    public static TheoryData<LatentArchitecture, int> StaticArchitectures => new()
    {
        { LatentArchitecture.Sd15, 4 },
        { LatentArchitecture.Sdxl, 4 },
        { LatentArchitecture.Sd3, 16 },
        { LatentArchitecture.Flux, 16 },
        { LatentArchitecture.Flux2, 32 },
        { LatentArchitecture.Chroma, 16 },
        { LatentArchitecture.AuraFlow, 4 },
        { LatentArchitecture.FLite, 16 },
        { LatentArchitecture.ZImage, 16 },
        { LatentArchitecture.Anima, 16 },
        { LatentArchitecture.Wan, 48 },
        { LatentArchitecture.Ltx, 128 },
        { LatentArchitecture.ChromaRadiance, 3 },
        { LatentArchitecture.ZetaChroma, 3 },
        { LatentArchitecture.HunyuanVideo, 16 },
        { LatentArchitecture.MiniMaxH3, 24 },
        { LatentArchitecture.HunyuanImage, 64 },
        { LatentArchitecture.MageFlow, 128 },
    };

    [Theory]
    [MemberData(nameof(StaticArchitectures))]
    public void RegisteredArchitecture_DecodesCanonicalLatent(LatentArchitecture architecture, int channels)
    {
        using Tensor latent = RandTensor([1, channels, 2, 3], seed: channels);
        byte[]? rgb = LatentPreview.DecodeLatent2Rgb(latent, architecture, out int width, out int height);
        Assert.NotNull(rgb);
        Assert.Equal(3, width);
        Assert.Equal(2, height);
        Assert.Equal(18, rgb!.Length);
    }

    [Fact]
    public void WanAndLtx_AreSupported()
    {
        Assert.True(LatentPreview.IsSupported(LatentArchitecture.Wan));
        Assert.True(LatentPreview.IsSupported(LatentArchitecture.Ltx));
    }

    [Fact]
    public void Wan_Rank5Latent_DecodesMiddleFrame()
    {
        Tensor latent = RandTensor([1, 48, 5, 4, 6], seed: 7);
        byte[]? rgb = LatentPreview.DecodeLatent2Rgb(latent, LatentArchitecture.Wan, out int w, out int h);
        Assert.NotNull(rgb);
        Assert.Equal(6, w);
        Assert.Equal(4, h);
        Assert.Equal(6 * 4 * 3, rgb!.Length);
        latent.Dispose();
    }

    [Fact]
    public void Wan21_Rank5Latent_DecodesEveryFrame()
    {
        using Tensor latent = RandTensor([1, 16, 5, 4, 6], seed: 8);
        byte[][]? frames = LatentPreview.DecodeVideoLatent2RgbFrames(
            latent, LatentArchitecture.Wan, out int w, out int h);
        Assert.NotNull(frames);
        Assert.Equal(5, frames!.Length);
        Assert.All(frames, frame => Assert.Equal(6 * 4 * 3, frame.Length));
        Assert.Equal(6, w);
        Assert.Equal(4, h);
    }

    [Fact]
    public void Ltx_Rank4Slice_Decodes()
    {
        Tensor latent = RandTensor([1, 128, 4, 6], seed: 9);
        byte[]? rgb = LatentPreview.DecodeLatent2Rgb(latent, LatentArchitecture.Ltx, out int w, out int h);
        Assert.NotNull(rgb);
        Assert.Equal(6, w);
        Assert.Equal(4, h);
        latent.Dispose();
    }

    [Fact]
    public void Wan_WrongChannelCount_ReturnsNull()
    {
        Tensor latent = RandTensor([1, 12, 5, 4, 6], seed: 3);
        byte[]? rgb = LatentPreview.DecodeLatent2Rgb(latent, LatentArchitecture.Wan, out _, out _);
        Assert.Null(rgb);
        latent.Dispose();
    }

    private static Tensor RandTensor(long[] shape, int seed)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
