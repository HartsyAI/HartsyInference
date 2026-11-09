using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Tests.Models.HairDeck;

public sealed class HairDeckOptionsTests
{
    [Fact]
    public void Defaults_MatchTheReferenceAndValidate()
    {
        HairDeckOptions options = new HairDeckOptions().Validate();

        Assert.Equal(50, options.Steps);
        Assert.Equal(2.0, options.Cfg);
        Assert.Equal(1, options.Samples);
        Assert.True(options.Colors);
        Assert.True(options.Chains);
        Assert.False(options.Rigged);
    }

    [Fact]
    public void Validate_RejectsMoreThanOneSampleInsteadOfDroppingIt()
    {
        HairDeckOptions options = new() { Samples = 2 };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => options.Validate());

        Assert.Contains("2 samples", error.Message);
    }

    [Fact]
    public void Validate_RejectsRiggedOutputWithoutChains()
    {
        Assert.Throws<NotSupportedException>(() => new HairDeckOptions { Rigged = true, Chains = false }.Validate());
    }

    [Fact]
    public void Validate_AcceptsRiggedOutputWithChains()
    {
        HairDeckOptions options = new HairDeckOptions { Rigged = true }.Validate();
        Assert.True(options.Rigged);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Validate_RejectsANonPositiveStepCount(int steps)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HairDeckOptions { Steps = steps }.Validate());
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Validate_RejectsAGuidanceScaleThatIsNotPositiveAndFinite(double cfg)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HairDeckOptions { Cfg = cfg }.Validate());
    }
}
