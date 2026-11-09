using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaRopeTests
{
    [Fact]
    public void InvFrequencies_AreRopeBaseToTheNegativeFractionOfTheChunk()
    {
        float[] freq = UtoniaRope.InvFrequencies(8, 16f);

        Assert.Equal(4, freq.Length);
        double[] expected = [1.0, 0.5, 0.25, 0.125];
        for (int j = 0; j < expected.Length; j++)
        {
            Assert.Equal(expected[j], freq[j], 6);
        }
    }
}
