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
}
