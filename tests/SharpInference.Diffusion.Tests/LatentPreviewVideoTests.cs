using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Utilities;

namespace SharpInference.Diffusion.Tests;

/// <summary>Latent2rgb preview coverage for the video architectures: Wan2.2 decodes the middle frame of a
/// rank-5 <c>[1, 48, T, H, W]</c> latent; LTX receives a rank-4 single-frame slice (its pipeline unpacks the
/// token-packed latent before the callback). Factors are Comfy's published tables — numeric hue accuracy is
/// inherently approximate; these tests verify wiring, shapes, and finite output.</summary>
public unsafe class LatentPreviewVideoTests
{
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
        Tensor latent = RandTensor([1, 16, 5, 4, 6], seed: 3);
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
