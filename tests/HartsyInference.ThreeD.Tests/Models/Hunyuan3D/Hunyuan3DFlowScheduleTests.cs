using HartsyInference.ThreeD.Models.Hunyuan3D;
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

    [Fact]
    public void ConsistencySigmas_MatchTheUpstreamTableForFiveSteps()
    {
        // numpy: linspace(0, 1, 1000)[[0] + (round(arange(1, 100) * 10) - 1)][floor(linspace(0, 100, 5, endpoint=False))], as float32
        float[] expected = [0f, 0.19919919967651367f, 0.39939939975738525f, 0.5995995998382568f, 0.7997997999191284f, 1f];

        Assert.Equal(expected, Hunyuan3DFlowSchedule.ConsistencySigmas(5, 100));
    }

    [Fact]
    public void ConsistencySigmas_MatchTheUpstreamTableForTenSteps()
    {
        float[] expected =
        [
            0f, 0.09909909963607788f, 0.19919919967651367f, 0.29929929971694946f, 0.39939939975738525f,
            0.49949949979782104f, 0.5995995998382568f, 0.6996996998786926f, 0.7997997999191284f, 0.8998998999595642f, 1f,
        ];

        Assert.Equal(expected, Hunyuan3DFlowSchedule.ConsistencySigmas(10, 100));
    }

    [Fact]
    public void Sigmas_SelectTheConsistencyTableByScheduler()
    {
        Assert.Equal(
            Hunyuan3DFlowSchedule.ConsistencySigmas(5, 100),
            Hunyuan3DFlowSchedule.Sigmas(5, Hunyuan3DScheduler.ConsistencyFlowMatchEuler, 100));
        Assert.Equal(Hunyuan3DFlowSchedule.Sigmas(5), Hunyuan3DFlowSchedule.Sigmas(5, Hunyuan3DScheduler.FlowMatchEuler));
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
            Assert.Equal(sigma, t);
            Assert.InRange(t, 0f, 1f);
        }
    }
}
