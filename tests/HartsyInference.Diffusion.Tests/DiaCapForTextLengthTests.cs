using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit coverage for <c>DiaTtsModel.Session.CapForTextLength</c> — the floor, the factor, and the one
/// property the short-prompt fix depends on: it never raises <c>requested</c>, only ever tightens it. See PR
/// #230 for the empirical evidence behind <see cref="DiaTtsModel.Session.FramesPerChar"/> and the 200-frame
/// floor. The floor/factor boundary cases below derive their <c>textLength</c> from the live constants (not a
/// hardcoded literal), so retuning <c>FramesPerChar</c> can't silently break them the way the first version of
/// this file did.</summary>
public sealed class DiaCapForTextLengthTests
{
    private const int MinFrames = DiaTtsModel.Session.MinFrames;
    private const int FramesPerChar = DiaTtsModel.Session.FramesPerChar;

    /// <summary>The largest textLength whose factor estimate (<c>textLength * FramesPerChar</c>) still falls
    /// at or under <see cref="MinFrames"/> -- i.e. the floor/factor boundary, from below.</summary>
    private const int LargestFloorLength = MinFrames / FramesPerChar;

    /// <summary>The largest textLength whose factor estimate still falls at or under the 1720 default --
    /// i.e. the factor/default-ceiling boundary, from below. A textLength here uses the factor, not the
    /// floor (<see cref="LargestFloorLength"/> < this, for any floor/factor/ceiling combination where the
    /// floor is reachable by the factor at all) and not the ceiling (<see cref="NeverRaisesRequested"/>
    /// covers that edge separately).</summary>
    private const int LargestMidRangeLength = DiaTtsModel.Session.DefaultMaxTokens / FramesPerChar;

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [MemberData(nameof(FloorBoundaryLength))]
    public void ShortTextHitsTheFloor_NotTheFactor(int textLength)
    {
        int result = DiaTtsModel.Session.CapForTextLength(requested: DiaTtsModel.Session.DefaultMaxTokens, textLength);

        Assert.Equal(MinFrames, result);
    }

    public static IEnumerable<object[]> FloorBoundaryLength()
    {
        yield return new object[] { LargestFloorLength };
    }

    [Theory]
    [MemberData(nameof(JustOverFloorLength))]
    [MemberData(nameof(MidRangeLength))]
    public void MidLengthTextUsesTheFramesPerCharFactor(int textLength)
    {
        int expected = textLength * FramesPerChar;

        int result = DiaTtsModel.Session.CapForTextLength(requested: DiaTtsModel.Session.DefaultMaxTokens, textLength);

        Assert.Equal(expected, result);
        Assert.True(expected > MinFrames, "test is only meaningful above the floor.");
        Assert.True(expected < DiaTtsModel.Session.DefaultMaxTokens, "test is only meaningful below the default requested ceiling.");
    }

    public static IEnumerable<object[]> JustOverFloorLength()
    {
        yield return new object[] { LargestFloorLength + 1 };
    }

    public static IEnumerable<object[]> MidRangeLength()
    {
        // The midpoint between the two boundaries, so it stays comfortably mid-range for any
        // floor/factor combination rather than sitting right against either edge.
        yield return new object[] { (LargestFloorLength + LargestMidRangeLength) / 2 };
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
    public void RequestedBelowFloorStillWins()
    {
        // requested below the floor: Math.Min still applies, so requested (not the floor) wins -- this is the
        // same "never raises requested" guarantee, restated for the floor branch specifically.
        int result = DiaTtsModel.Session.CapForTextLength(requested: 50, textLength: 1);

        Assert.Equal(50, result);
    }
}
