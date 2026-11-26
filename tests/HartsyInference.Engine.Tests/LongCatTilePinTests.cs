using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat generation pins its attention tile height through the per-request knob profile, for its own scope only, and keeps the
/// request's other knobs and any sdpaTile the caller already set.</summary>
public sealed class LongCatTilePinTests
{
    [Fact]
    public void TheTileIsPinnedForTheScopeAndReleasedAfterwards()
    {
        Assert.Null(EngineKnobs.SdpaTile.Value);
        long expected = (512L << 20) / (32L * 4600 * sizeof(float));
        using (LongCatRecipeSupport.PinAttentionTile(heads: 32, keyTokens: 4600))
        {
            Assert.Equal((int)expected, EngineKnobs.SdpaTile.Value);
        }
        Assert.Null(EngineKnobs.SdpaTile.Value);
    }

    [Fact]
    public void AHugeSequenceStillGetsAtLeastOneRow()
    {
        using (LongCatRecipeSupport.PinAttentionTile(heads: 32, keyTokens: 1L << 40))
        {
            Assert.Equal(1, EngineKnobs.SdpaTile.Value);
        }
    }

    [Fact]
    public void TheRequestsOtherKnobsSurviveThePin()
    {
        using IDisposable outer = KnobProfile.Create("request").With(EngineKnobs.Int8NChunk, 4096).Push();
        using (LongCatRecipeSupport.PinAttentionTile(heads: 32, keyTokens: 4600))
        {
            Assert.Equal(4096, EngineKnobs.Int8NChunk.Value);
            Assert.NotNull(EngineKnobs.SdpaTile.Value);
        }
        Assert.Equal(4096, EngineKnobs.Int8NChunk.Value);
        Assert.Null(EngineKnobs.SdpaTile.Value);
    }

    [Fact]
    public void ACallerWhoSetTheTileKeepsIt()
    {
        using IDisposable outer = KnobProfile.Create("request").With(EngineKnobs.SdpaTile, 7).Push();
        using (LongCatRecipeSupport.PinAttentionTile(heads: 32, keyTokens: 4600))
        {
            Assert.Equal(7, EngineKnobs.SdpaTile.Value);
        }
    }
}
