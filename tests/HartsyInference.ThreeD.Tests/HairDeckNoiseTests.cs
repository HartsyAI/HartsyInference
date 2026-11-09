using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairDeckNoiseTests
{
    [Fact]
    public void Create_IsDeterministicPerSeedAndStandardNormal()
    {
        using Tensor a = HairDeckNoise.Create(7, 200, 200);
        using Tensor b = HairDeckNoise.Create(7, 200, 200);
        using Tensor c = HairDeckNoise.Create(8, 200, 200);
        float[] va = HairDeckNumerics.ToArray(a), vb = HairDeckNumerics.ToArray(b), vc = HairDeckNumerics.ToArray(c);

        Assert.Equal(va, vb);
        Assert.NotEqual(va, vc);
        double mean = va.Average(v => (double)v);
        double variance = va.Select(v => (v - mean) * (v - mean)).Average();
        Assert.InRange(mean, -0.02, 0.02);
        Assert.InRange(Math.Sqrt(variance), 0.97, 1.03);
    }
}
