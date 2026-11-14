using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The noise seed keeps the full request seed, so a seed at or above 2^31 seeds the noise the way torch.manual_seed
/// does. Only a negative or unset seed takes the 31-bit fallback.</summary>
public sealed class MiniMaxH3NoiseSeedTests
{
    [Theory]
    [InlineData(-1L, 42, 42L)]
    [InlineData(0L, 42, 0L)]
    [InlineData(5L, 42, 5L)]
    [InlineData(2147483647L, 42, 2147483647L)]
    [InlineData(2147483648L, 42, 2147483648L)]
    [InlineData(3000000000L, 42, 3000000000L)]
    [InlineData(4294967303L, 42, 4294967303L)]
    public void NoiseSeedKeepsTheFullRequestSeedAndFallsBackOnlyForNegativeSeeds(long requestSeed, int fallback, long expected)
    {
        Assert.Equal(expected, MiniMaxH3ImageRecipePipeline.NoiseSeed(requestSeed, fallback));
    }
}
