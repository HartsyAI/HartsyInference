using HartsyInference.Audio.Models.F5Tts;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Unit tests for the F5 duration heuristic. The expectations are upstream F5's mel lengths for the sweep's
/// reference clip: 341 reference frames and 66 bytes of reference text (the ". " plus upstream's extra space).</summary>
public sealed class F5DurationTests
{
    [Theory]
    [InlineData(0, 0.3)]
    [InlineData(9, 0.3)]
    [InlineData(10, 1.0)]
    public void LocalSpeed_ShortGenerationsUseShortTextSpeed(int genBytes, double expected)
        => Assert.Equal(expected, F5Duration.LocalSpeed(genBytes, 1.0));

    [Fact]
    public void LocalSpeed_LongGenerationsKeepUserSpeed()
        => Assert.Equal(0.8, F5Duration.LocalSpeed(51, 0.8));

    [Theory]
    [InlineData(4, 68)]
    [InlineData(14, 72)]
    [InlineData(51, 263)]
    public void TargetFrames_MatchUpstreamOnTheSweepReference(int genBytes, int expected)
        => Assert.Equal(expected, F5Duration.TargetFrames(341, 66, genBytes, 1.0));

    [Fact]
    public void TargetFrames_EmptyReferenceDividesByOne()
        => Assert.Equal(341 * 51, F5Duration.TargetFrames(341, 0, 51, 1.0));
}
