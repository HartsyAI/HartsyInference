using HartsyInference.Audio.Models.Auk;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Unit tests for AuK duration resolution and frame counting.</summary>
public sealed class AukDurationTests
{
    [Theory]
    [InlineData(1.0, 50)]
    [InlineData(0.001, 1)]
    [InlineData(0.0, 1)]
    [InlineData(2.01, 101)]
    public void Frames_IsCeilOfSecondsTimesRateOverHop(double seconds, int expected) => Assert.Equal(expected, AukDuration.Frames(seconds));

    [Fact]
    public void ExplicitSeconds_Wins()
        => Assert.Equal(3.5, AukDuration.ResolveSeconds(3.5, "abc", "de", 10.0));

    [Fact]
    public void Heuristic_UsesUtf8ByteRatioAndSpeed()
    {
        // "héllo" = 6 bytes, "ab" = 2 bytes
        Assert.Equal(2.0 * 6 / 2 / 1.5, AukDuration.ResolveSeconds(null, "héllo", "ab", 2.0, 1.5), 12);
    }

    [Fact]
    public void NoTexts_DefaultsToReferenceLength()
        => Assert.Equal(4.0, AukDuration.ResolveSeconds(null, null, null, 4.0));

    [Fact]
    public void InstructWithoutReferenceOrDuration_Throws()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => AukDuration.ResolveSeconds(null, "hi", null, null));
        Assert.Contains("explicit duration", ex.Message);
    }

    [Fact]
    public void ResolveFrames_ComposesBoth() => Assert.Equal(100, AukDuration.ResolveFrames(2.0, null, null, null));
}
