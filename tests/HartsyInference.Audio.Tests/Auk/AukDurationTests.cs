using HartsyInference.Audio.Models.Auk;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Unit tests for AuK duration resolution and frame counting.</summary>
public sealed class AukDurationTests
{
    [Theory]
    [InlineData(1.0, 50)]
    public void Frames_IsCeilOfSecondsTimesRateOverHop(double seconds, int expected) => Assert.Equal(expected, AukDuration.Frames(seconds));

    [Fact]
    public void ExplicitSeconds_Wins()
        => Assert.Equal(3.5, AukDuration.ResolveSeconds(3.5, "abc", "de", 10.0));

    [Fact]
    public void InstructWithoutReferenceOrDuration_Throws()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => AukDuration.ResolveSeconds(null, "hi", null, null));
        Assert.Contains("explicit duration", ex.Message);
    }

    [Theory]
    [InlineData(0.0)]
    public void ExplicitSeconds_MustBePositiveEvenWithAReferenceClip(double seconds)
        => Assert.Throws<ArgumentOutOfRangeException>(() => AukDuration.ResolveSeconds(seconds, "a", "b", 10.0));

}
