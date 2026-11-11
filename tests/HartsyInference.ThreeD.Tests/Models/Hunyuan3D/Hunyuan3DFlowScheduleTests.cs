using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DFlowScheduleTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(20)]
    [InlineData(50)]
    public void Sigmas_AscendFromZeroToOneWithATrailingOne(int steps)
    {
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(steps);

        Assert.Equal(steps + 1, sigmas.Length);
        Assert.Equal(0f, sigmas[0]);
        Assert.Equal(1f, sigmas[steps - 1]);
        Assert.Equal(1f, sigmas[steps]);
        for (int i = 1; i < sigmas.Length; i++)
            Assert.True(sigmas[i] >= sigmas[i - 1]);
    }

    [Fact]
    public void Sigmas_AreEvenlySpacedLikeNumpyLinspace()
    {
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(5);

        Assert.Equal([0f, 0.25f, 0.5f, 0.75f, 1f, 1f], sigmas);
    }

    [Fact]
    public void Sigmas_ForOneStepIntegrateTheWholeIntervalFromZero()
    {
        Assert.Equal([0f, 1f], Hunyuan3DFlowSchedule.Sigmas(1));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Sigmas_RejectANonPositiveStepCount(int steps)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Hunyuan3DFlowSchedule.Sigmas(steps));
    }

    [Fact]
    public void DitTimestep_StaysInUnitRangeBecauseTheDitScalesByTimeFactorItself()
    {
        foreach (float sigma in Hunyuan3DFlowSchedule.Sigmas(50))
        {
            float t = Hunyuan3DFlowSchedule.DitTimestep(sigma);
            Assert.InRange(t, 0f, 1f);
        }
    }
}
