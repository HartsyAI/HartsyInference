using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Segment crop padding: 0 means the SwarmUI default unless the caller marks it exact, where 0 crops tight.</summary>
public sealed class SegmentOversizeTests
{
    [Theory]
    [InlineData(0, false, 16)]
    [InlineData(0, true, 0)]
    [InlineData(24, true, 24)]
    public void ResolveOversize_AppliesTheDefaultOnlyWhenNotExact(int oversize, bool exact, int expected)
    {
        Regional regional = new Regional { MaskOversize = oversize, ExactMaskOversize = exact };

        Assert.Equal(expected, SegmentRefinement.ResolveOversize(regional));
    }
}
