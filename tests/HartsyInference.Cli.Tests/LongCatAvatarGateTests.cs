using HartsyInference.Cli.Commands;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The video command's LongCat-Video-Avatar gates: a 1.6 request is refused before any download, and the likeness consent line prints on
/// every non-quiet run that resolves to the avatar recipe, whether it names the catalog row or the family token.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarGateTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("1.5")]
    public void TheDefaultAndNamed15RequestsAreNotRefused(string? variant)
    {
        Assert.Null(LongCatAvatarGate.Refusal("longcat-avatar", variant));
    }

    [Theory]
    [InlineData("1.6")]
    [InlineData("avatar-1.6")]
    public void A16BuildRequestIsRefusedWithItsReason(string variant)
    {
        string? refusal = LongCatAvatarGate.Refusal("longcat-avatar", variant);

        Assert.NotNull(refusal);
        Assert.Contains("1.6 INT8 catalog row", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void TheRefusalIsOnlyForTheAvatarFamily()
    {
        Assert.Null(LongCatAvatarGate.Refusal("ltx-video", "1.6"));
    }

    [Fact]
    public void TheConsentLinePrintsOnANonQuietRunAndIsSilentUnderQuiet()
    {
        Assert.Equal(LongCatAvatarGate.ConsentText, LongCatAvatarGate.ConsentNotice("longcat-avatar", quiet: false));
        Assert.Null(LongCatAvatarGate.ConsentNotice("longcat-avatar", quiet: true));
    }

    [Fact]
    public void TheConsentLineIsSilentForOtherFamiliesAndForNoFamily()
    {
        Assert.Null(LongCatAvatarGate.ConsentNotice("ltx-video", quiet: false));
        Assert.Null(LongCatAvatarGate.ConsentNotice(null, quiet: false));
    }
}
