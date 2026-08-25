using HartsyInference.Core.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Pins the OOM escalation ladder: each rung must give up strictly more than the last, stop at the top, and never overrule a lever the caller pinned on purpose.</summary>
public sealed class VramEscalationTests
{
    /// <summary>Ordered by how much is surrendered, NOT by the enum's numeric order — Performance is 0 but is the
    /// least aggressive, and Auto re-enters at Balanced because its measurement already ran and was not enough.</summary>
    [Theory]
    [InlineData(VramTier.Performance, VramTier.Balanced)]
    [InlineData(VramTier.Auto, VramTier.Balanced)]
    [InlineData(VramTier.Balanced, VramTier.Aggressive)]
    [InlineData(VramTier.Aggressive, VramTier.Maximum)]
    public void EachRungGivesUpMore(VramTier from, VramTier expected)
        => Assert.Equal(expected, VramPolicyResolver.Escalate(from));

    /// <summary>The ladder must terminate, or a genuinely too-large request retries forever instead of failing.</summary>
    [Fact]
    public void MaximumIsTheTop()
    {
        Assert.Null(VramPolicyResolver.Escalate(VramTier.Maximum));
        Assert.Null(VramPolicyResolver.Escalate(VramPolicy.For(VramTier.Maximum)));
    }

    /// <summary>Walking from any start reaches Maximum in a bounded number of steps.</summary>
    [Theory]
    [InlineData(VramTier.Performance)]
    [InlineData(VramTier.Auto)]
    [InlineData(VramTier.Balanced)]
    [InlineData(VramTier.Aggressive)]
    public void TheLadderTerminates(VramTier start)
    {
        VramTier? tier = start;
        int rungs = 0;
        while (tier is VramTier t && VramPolicyResolver.Escalate(t) is VramTier next)
        {
            tier = next;
            Assert.True(++rungs < 10, "escalation did not terminate");
        }
        Assert.Equal(VramTier.Maximum, tier);
    }

    [Fact]
    public void EscalatingActuallyTurnsStreamingOn()
    {
        VramPolicy balanced = VramPolicy.For(VramTier.Balanced);
        Assert.Equal(LeverState.Auto, balanced.WeightStreaming);

        VramPolicy? harder = VramPolicyResolver.Escalate(balanced);
        Assert.NotNull(harder);
        Assert.Equal(VramTier.Aggressive, harder!.Tier);
        Assert.Equal(LeverState.On, harder.WeightStreaming);
    }

    /// <summary>A lever the caller pinned survives the escalation. An automatic retry is not the place to overrule
    /// an explicit choice — most sharply for streaming pinned Off, where the operator asked for a loud failure and
    /// must not be handed a silent slow success instead.</summary>
    [Fact]
    public void PinnedLeversSurviveEscalation()
    {
        VramPolicy pinned = VramPolicy.For(VramTier.Balanced) with { WeightStreaming = LeverState.Off };
        VramPolicy harder = VramPolicyResolver.Escalate(pinned)!;

        Assert.Equal(VramTier.Aggressive, harder.Tier);
        Assert.Equal(LeverState.Off, harder.WeightStreaming);
        // Everything the caller did NOT pin still hardens.
        Assert.Equal(CachePrecision.Half, harder.Caches);
    }

    /// <summary>Explicit numeric tuning is carried across too, rather than silently reset by the retry.</summary>
    [Fact]
    public void ExplicitBudgetsCarryAcross()
    {
        VramPolicy tuned = VramPolicy.For(VramTier.Auto) with { PrefetchAhead = 1, HeadroomBytes = 4096 };
        VramPolicy harder = VramPolicyResolver.Escalate(tuned)!;
        Assert.Equal(1, harder.PrefetchAhead);
        Assert.Equal(4096, harder.HeadroomBytes);
    }
}
