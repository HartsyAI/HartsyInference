using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Utonia;

public sealed class UtoniaPatchPlanTests
{
    [Fact]
    public void Create_PadsLongSampleByCopyingPreviousPatchTail_AndKeepsShortSampleUnpadded()
    {
        // Sample 0 has 5 points with patch 4, so it pads to 8 slots; sample 1 has 2 points, shorter than a patch.
        UtoniaPatchPlan plan = UtoniaPatchPlan.Create(new[] { 5, 2 }, 4);

        Assert.Equal(new[] { 0, 1, 2, 3, 4, 1, 2, 3, 5, 6 }, plan.Pad);
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 8, 9 }, plan.Unpad);
        Assert.Equal(new[] { 0, 4, 8 }, plan.SegmentStart);
        Assert.Equal(new[] { 4, 4, 2 }, plan.SegmentLength);
    }

    [Fact]
    public void Create_FullPatchesStayIdentityWithOneSegmentPerPatch()
    {
        UtoniaPatchPlan plan = UtoniaPatchPlan.Create(new[] { 8 }, 4);

        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, plan.Pad);
        Assert.Equal(new[] { 0, 4 }, plan.SegmentStart);
        Assert.Equal(new[] { 4, 4 }, plan.SegmentLength);
    }

    [Fact]
    public void Create_SkipsEmptySamplesWithoutSegments()
    {
        UtoniaPatchPlan plan = UtoniaPatchPlan.Create(new[] { 0, 3 }, 4);

        Assert.Equal(new[] { 0, 1, 2 }, plan.Pad);
        Assert.Equal(new[] { 0, 1, 2 }, plan.Unpad);
        Assert.Equal(new[] { 0 }, plan.SegmentStart);
        Assert.Equal(new[] { 3 }, plan.SegmentLength);
    }

    [Fact]
    public void Create_RejectsNonPositivePatchSize()
    {
        Assert.Throws<HartsyInferenceException>(() => UtoniaPatchPlan.Create(new[] { 3 }, 0));
    }
}
