using HartsyInference.Cli.Commands;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The video command's LongCat-Video-Avatar gates: a 1.6 request is routed to the 1.6 INT8 catalog row before any download, and the likeness
/// consent line prints on every non-quiet run that resolves to the avatar recipe, whether it names the catalog row or the family token.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarGateTests
{
    [Theory]
    [InlineData("longcat-avatar", "longcat-avatar")]
    [InlineData("longcat-avatar:1.5", "longcat-avatar:1.5")]
    [InlineData("longcat-avatar-1.6", "longcat-avatar-1.6")]
    public void ANonSixPointOneRequestIsUnchanged(string model, string expected)
    {
        Assert.Equal(expected, LongCatAvatarGate.ModelName(model));
    }

    [Theory]
    [InlineData("longcat-avatar:1.6")]
    [InlineData("longcat-avatar:avatar-1.6")]
    public void A16BuildIsRoutedToTheInt8Row(string model)
    {
        Assert.Equal("longcat-avatar-1.6", LongCatAvatarGate.ModelName(model));
    }

    [Fact]
    public void TheRoutingIsOnlyForTheAvatarFamily()
    {
        Assert.Equal("ltx-video:1.6", LongCatAvatarGate.ModelName("ltx-video:1.6"));
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
