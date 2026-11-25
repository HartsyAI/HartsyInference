using HartsyInference.Core.Layers.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The segment lists of LongCat's conditioning-aware attention, with the numbers worked by hand from the reference's slicing
/// (<c>Attention.forward</c> in LongCat-Video @ 58eb336: <c>start = ref - range - cond + refCount</c>, <c>end = ref + range -
/// cond + refCount + 1</c>).</summary>
public sealed class ConditionedAttentionPlanTests
{
    [Fact]
    public void NoConditioningIsOneDenseSegment()
    {
        Assert.Equal([new AttentionSegment(0, 12, 0, 12)], ConditionedAttentionPlan.ForConditionedFrames(3, 4, 0));
    }

    [Fact]
    public void ConditioningFramesAttendOnlyThemselvesAndNoisyFramesAttendAll()
    {
        Assert.Equal([new AttentionSegment(0, 3, 0, 3), new AttentionSegment(3, 9, 0, 12)], ConditionedAttentionPlan.ForConditionedFrames(3, 4, 1));
    }

    [Fact]
    public void AllConditioningHasNoNoisySegment()
    {
        Assert.Equal([new AttentionSegment(0, 12, 0, 12)], ConditionedAttentionPlan.ForConditionedFrames(3, 4, 4));
    }

    [Fact]
    public void ReferenceWithSingleConditioningFrameIsTheBasePlan()
    {
        Assert.Equal(ConditionedAttentionPlan.ForConditionedFrames(3, 4, 1), ConditionedAttentionPlan.ForReferencedFrames(3, 4, 1, 1, 10, 3));
    }

    [Fact]
    public void ReferenceAndContextAreSeparateSelfOnlyGroups()
    {
        AttentionSegment[] plan = ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, 4, 0);

        Assert.Equal(
            [new AttentionSegment(0, 2, 0, 2), new AttentionSegment(2, 4, 2, 4), new AttentionSegment(6, 12, 0, 18)], plan);
    }

    [Fact]
    public void NoisyFramesNearTheReferenceIndexSkipTheReference()
    {
        // ref 4, range 1, 3 conditioning frames, 9 frames: skipped noisy frames are [1, 4) of the 6 noisy frames.
        AttentionSegment[] plan = ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, 4, 1);

        Assert.Equal(
        [
            new AttentionSegment(0, 2, 0, 2), new AttentionSegment(2, 4, 2, 4), new AttentionSegment(6, 2, 0, 18),
            new AttentionSegment(8, 6, 2, 16), new AttentionSegment(14, 4, 0, 18),
        ], plan);
    }

    [Fact]
    public void SkipWindowStartingAtTheFirstNoisyFrameHasNoFrontSegment()
    {
        // ref 3, range 1, 3 conditioning frames: window [0, 3), so noisy frames 0..2 skip and 3..5 do not.
        AttentionSegment[] plan = ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, 3, 1);

        Assert.Equal(
        [
            new AttentionSegment(0, 2, 0, 2), new AttentionSegment(2, 4, 2, 4), new AttentionSegment(6, 6, 2, 16), new AttentionSegment(12, 6, 0, 18),
        ], plan);
    }

    [Theory]
    [InlineData(10, 1)]   // window [7, 10) runs past the 6 noisy frames
    [InlineData(1, 2)]    // window starts before the first noisy frame
    [InlineData(4, 0)]    // range 0 disables the skip
    public void SkipWindowThatDoesNotFitIsIgnored(int refIndex, int range)
    {
        AttentionSegment[] plan = ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 1, refIndex, range);

        Assert.Equal(new AttentionSegment(6, 12, 0, 18), plan[^1]);
        Assert.Equal(3, plan.Length);
    }

    [Fact]
    public void EveryPlanTilesTheQueryRows()
    {
        foreach (AttentionSegment[] plan in new[]
        {
            ConditionedAttentionPlan.ForReferencedFrames(3, 11, 4, 1, 5, 2), ConditionedAttentionPlan.ForReferencedFrames(1, 8, 2, 1, 2, 1),
            ConditionedAttentionPlan.ForConditionedFrames(5, 7, 2),
        })
        {
            int tokens = plan.Sum(static s => s.QueryCount);
            AttentionSegment.Validate(plan, tokens);
        }
    }

    [Fact]
    public void MultiFrameContextNeedsAReferenceSlot()
    {
        Assert.Throws<ArgumentException>(() => ConditionedAttentionPlan.ForReferencedFrames(2, 9, 3, 0, 4, 1));
    }

    [Fact]
    public void ValidateRejectsGapsAndOutOfRangeKeys()
    {
        Assert.Throws<ArgumentException>(() => AttentionSegment.Validate([new AttentionSegment(0, 2, 0, 2), new AttentionSegment(3, 3, 0, 6)], 6));
        Assert.Throws<ArgumentException>(() => AttentionSegment.Validate([new AttentionSegment(0, 6, 1, 6)], 6));
        Assert.Throws<ArgumentException>(() => AttentionSegment.Validate([new AttentionSegment(0, 4, 0, 4)], 6));
    }
}
