using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class VideoControlValidationTests
{
    [Fact]
    public void StrengthMustBeFiniteNonNegativeAndRepresentableAsF32()
    {
        Assert.True(VideoControlValidation.IsValidStrength(0.0));
        Assert.True(VideoControlValidation.IsValidStrength(float.MaxValue));
        Assert.False(VideoControlValidation.IsValidStrength(-double.Epsilon));
        Assert.False(VideoControlValidation.IsValidStrength((double)float.MaxValue * 2.0));
        Assert.False(VideoControlValidation.IsValidStrength(double.NaN));
    }

    [Fact]
    public void WindowIsInclusiveOrderedAndNormalized()
    {
        Assert.True(VideoControlValidation.IsValidWindow(0.0, 1.0));
        Assert.True(VideoControlValidation.IsValidWindow(0.5, 0.5));
        Assert.False(VideoControlValidation.IsValidWindow(-double.Epsilon, 1.0));
        Assert.False(VideoControlValidation.IsValidWindow(0.75, 0.25));
        Assert.False(VideoControlValidation.IsValidWindow(0.0, double.PositiveInfinity));
    }
}
