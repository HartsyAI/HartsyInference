using HartsyInference.Audio.Models.Auk;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Unit tests for the AuK sampling schedule: Flash fixed grid, base sway grid, Euler and CFG span helpers.</summary>
public sealed class AukScheduleTests
{
    [Fact]
    public void Flash_UsesFixedFourStepGridWithoutCfg()
    {
        AukSchedule s = AukSchedule.Flash();
        Assert.Equal(4, s.Steps);
        Assert.False(s.UsesCfg);
        Assert.Equal([0f, 0.07612049579620361f, 0.2928932309150696f, 0.6173166036605835f, 1f], s.Timesteps);
        for (int i = 0; i < 4; i++) Assert.Equal(s.Timesteps[i + 1] - s.Timesteps[i], s.Deltas[i]);
        Assert.Equal(1f, s.Timesteps.Zip(s.Deltas, (t, d) => t + d).Last(), 5);
    }

    [Fact]
    public void Base_MatchesSwayFormulaInDouble()
    {
        AukSchedule s = AukSchedule.Base();
        Assert.Equal(32, s.Steps);
        Assert.Equal(2f, s.Cfg);
        Assert.True(s.UsesCfg);
        for (int i = 0; i <= 32; i++)
        {
            double t = i / 32.0;
            double expected = t + -1.0 * (Math.Cos(Math.PI / 2 * t) - 1 + t);
            Assert.Equal(expected, s.Timesteps[i], 5);
        }
        Assert.Equal(0f, s.Timesteps[0]);
        Assert.Equal(1f, s.Timesteps[32], 5);
        Assert.All(s.Deltas, d => Assert.True(d > 0));
    }

    [Fact]
    public void EulerStep_AddsVelocityTimesDt()
    {
        float[] x = [1f, 2f, 3f];
        AukSchedule.EulerStep(x, [1f, -2f, 0.5f], 0.25f);
        Assert.Equal([1.25f, 1.5f, 3.125f], x);
        Assert.Throws<ArgumentException>(() => AukSchedule.EulerStep(new float[2], new float[3], 0.1f));
    }

    [Fact]
    public void CfgCombine_ExtrapolatesFromCondAwayFromUncond_AndSkipsTinyCfg()
    {
        float[] v = [1f, 4f];
        AukSchedule.CfgCombine(v, [0f, 5f], 2f);
        Assert.Equal([3f, 2f], v);
        float[] untouched = [1f, 4f];
        AukSchedule.CfgCombine(untouched, [0f, 5f], 1e-6f);
        Assert.Equal([1f, 4f], untouched);
    }
}
