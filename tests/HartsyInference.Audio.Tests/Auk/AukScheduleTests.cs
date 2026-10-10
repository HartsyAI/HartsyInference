using HartsyInference.Audio.Models.Auk;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Unit tests for the AuK sampling schedule: Flash fixed grid, base sway grid, Euler and CFG span helpers.</summary>
public sealed class AukScheduleTests
{
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
