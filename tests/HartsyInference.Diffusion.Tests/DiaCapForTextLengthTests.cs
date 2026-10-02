using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit coverage for <c>DiaTtsModel.Session.CapForTextLength</c> — the floor, the factor, and the one
/// property the short-prompt fix depends on: it never raises <c>requested</c>, only ever tightens it. See PR
/// #230 for the empirical evidence behind the 12 frames/char factor and the 200-frame floor.</summary>
public sealed class DiaCapForTextLengthTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(16)] // 16 * 12 = 192, still under the 200 floor.
    public void ShortTextHitsTheFloor_NotTheFactor(int textLength)
    {
        int result = DiaTtsModel.Session.CapForTextLength(requested: 1720, textLength);

        Assert.Equal(DiaTtsModel.Session.MinFrames, result);
    }

    [Theory]
    [InlineData(17)] // 17 * 12 = 204, just over the floor.
    [InlineData(100)]
    [InlineData(143)] // 143 * 12 = 1716, just under the 1720 default.
    public void MidLengthTextUsesTheFramesPerCharFactor(int textLength)
    {
        int expected = textLength * DiaTtsModel.Session.FramesPerChar;

        int result = DiaTtsModel.Session.CapForTextLength(requested: 1720, textLength);

        Assert.Equal(expected, result);
        Assert.True(expected > DiaTtsModel.Session.MinFrames, "test is only meaningful above the floor.");
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(1720, 1)]
    [InlineData(1720, 10_000)] // a huge prompt's own factor-derived estimate would dwarf 1720.
    [InlineData(50, 1000)] // factor/floor would compute 12_000 here -- requested must still win.
    public void NeverRaisesRequested(int requested, int textLength)
    {
        int result = DiaTtsModel.Session.CapForTextLength(requested, textLength);

        Assert.True(result <= requested, $"CapForTextLength({requested}, {textLength}) returned {result}, above the requested ceiling.");
    }

    [Fact]
    public void FloorWinsOverAnEvenSmallerRequestedValue()
    {
        // requested below the floor: Math.Min still applies, so requested (not the floor) wins -- this is the
        // same "never raises requested" guarantee, restated for the floor branch specifically.
        int result = DiaTtsModel.Session.CapForTextLength(requested: 50, textLength: 1);

        Assert.Equal(50, result);
    }
}
