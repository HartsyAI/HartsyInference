using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Tests;

/// <summary>Correctness tests for the reusable Wan2.2 VAE primitives — pixel-shuffle patchify/unpatchify and the 48-channel latent norm (no GPU/checkpoint).</summary>
public unsafe class Wan22VaePrimitivesTests
{
    [Fact]
    public void PatchifyUnpatchify_RoundTrips()
    {
        int b = 1, c = 3, t = 2, h = 8, w = 6, p = 2;
        Tensor x = Random([b, c, t, h, w], seed: 5);
        Tensor packed = Wan22VaePatch.Patchify(x, p);
        Assert.Equal(c * p * p, (int)packed.Shape[1]);
        Assert.Equal(h / p, (int)packed.Shape[3]);
        Assert.Equal(w / p, (int)packed.Shape[4]);

        Tensor back = Wan22VaePatch.Unpatchify(packed, p);
        AssertEqual(x, back, 0f);
    }

    [Fact]
    public void Patchify_PlacesPixelsAtExpectedChannels()
    {
        // 1×1×1×2×2 single block, channel 0. packed channel = c*p² + r*p + q.
        int p = 2;
        Tensor x = new Tensor(new TensorShape([1L, 1, 1, 2, 2]), DType.F32);
        float* xp = (float*)x.DataPointer;
        xp[0] = 10; // (q=0,r=0) → packed ch 0
        xp[1] = 11; // (q=0,r=1) → packed ch r*p+q = 2
        xp[2] = 12; // (q=1,r=0) → packed ch 1
        xp[3] = 13; // (q=1,r=1) → packed ch 3
        Tensor packed = Wan22VaePatch.Patchify(x, p);
        float* pp = (float*)packed.DataPointer; // shape [1,4,1,1,1]
        Assert.Equal(10f, pp[0]);
        Assert.Equal(12f, pp[1]);
        Assert.Equal(11f, pp[2]);
        Assert.Equal(13f, pp[3]);
    }

    [Fact]
    public void LatentNorm_DenormalizeIsInverseOfNormalize()
    {
        Tensor z = Random([1, Wan22VaeLatentNorm.Channels, 1, 2, 2], seed: 9);
        Tensor original = Clone(z);
        Wan22VaeLatentNorm.Normalize(z);
        Wan22VaeLatentNorm.Denormalize(z);
        AssertEqual(original, z, 1e-4f);
    }

    [Fact]
    public void LatentNorm_DenormalizeAppliesChannelStdMean()
    {
        Tensor z = new Tensor(new TensorShape([1L, Wan22VaeLatentNorm.Channels, 1, 1, 1]), DType.F32);
        float* p = (float*)z.DataPointer;
        for (int c = 0; c < Wan22VaeLatentNorm.Channels; c++) p[c] = 1.0f;
        Wan22VaeLatentNorm.Denormalize(z);
        for (int c = 0; c < Wan22VaeLatentNorm.Channels; c++)
            Assert.Equal(Wan22VaeLatentNorm.Std[c] + Wan22VaeLatentNorm.Mean[c], p[c], 4);
    }

    [Fact]
    public void LatentNorm_HasExactly48Constants()
    {
        Assert.Equal(48, Wan22VaeLatentNorm.Mean.Length);
        Assert.Equal(48, Wan22VaeLatentNorm.Std.Length);
    }

    private static Tensor Random(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Clone(Tensor s)
    {
        Tensor t = new Tensor(s.Shape, DType.F32);
        long n = s.Shape.ElementCount;
        Buffer.MemoryCopy(s.DataPointer, t.DataPointer, n * 4, n * 4);
        return t;
    }

    private static void AssertEqual(Tensor a, Tensor b, float tol)
    {
        long n = a.Shape.ElementCount;
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < n; i++)
            Assert.True(MathF.Abs(pa[i] - pb[i]) <= tol, $"idx {i}: {pa[i]} vs {pb[i]}");
    }
}
