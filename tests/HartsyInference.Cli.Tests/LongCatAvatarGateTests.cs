using HartsyInference.Cli.Commands;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The video command's LongCat-Video-Avatar consent gate: the likeness consent line prints on every non-quiet run that resolves to the avatar recipe.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarGateTests
{
    [Fact]
    public void TheConsentLinePrintsOnANonQuietRunAndIsSilentUnderQuiet()
    {
        Assert.Equal(LongCatAvatarGate.ConsentText, LongCatAvatarGate.ConsentNotice("longcat-avatar", quiet: false));
        Assert.Null(LongCatAvatarGate.ConsentNotice("longcat-avatar", quiet: true));
    }

    [Theory]
    [InlineData("longcat-avatar-int8")]
    [InlineData("longcat-avatar-1.6")]
    [InlineData("longcat-avatar-1.6-bf16")]
    public void TheConsentLinePrintsForEveryAvatarRow(string row) =>
        Assert.Equal(LongCatAvatarGate.ConsentText, LongCatAvatarGate.ConsentNotice(row, quiet: false));

    [Fact]
    public void TheConsentLineIsSilentForOtherFamiliesAndForNoFamily()
    {
        Assert.Null(LongCatAvatarGate.ConsentNotice("ltx-video", quiet: false));
        Assert.Null(LongCatAvatarGate.ConsentNotice(null, quiet: false));
    }
}
