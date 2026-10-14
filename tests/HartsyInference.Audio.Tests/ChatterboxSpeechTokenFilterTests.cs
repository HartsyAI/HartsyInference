using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Upstream <c>drop_invalid_tokens</c> + <c>speech_tokens &lt; 6561</c> before S3Gen.</summary>
public sealed class ChatterboxSpeechTokenFilterTests
{
    [Fact]
    public void DropsEverythingThroughStartTokenAndNonCodebookIds()
    {
        int[] kept = ChatterboxPipeline.DropInvalidSpeechTokens([5, 6561, 7, 6563, 8, 6560, 8193], 6561, 6561);
        Assert.Equal([7, 8, 6560], kept);
    }

    [Fact]
    public void WithoutStartToken_KeepsCodebookIdsOnly()
    {
        int[] kept = ChatterboxPipeline.DropInvalidSpeechTokens([0, 6600, 42], 6561, 6561);
        Assert.Equal([0, 42], kept);
    }
}
