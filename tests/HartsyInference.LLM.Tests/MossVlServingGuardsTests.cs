using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

/// <summary>The reply-length cap that keeps a MOSS-VL request from sizing its KV cache past the context window.</summary>
public sealed class MossVlServingGuardsTests
{
    [Theory]
    [InlineData(512, 100, 32_768, 512)]
    [InlineData(100_000, 100, 32_768, 32_668)]
    [InlineData(32_768, 32_767, 32_768, 1)]
    public void ReplyLengthIsCappedByTheContextLeft(int requested, int prompt, int context, int expected) =>
        Assert.Equal(expected, MossVlGenerator.ClampToContext(requested, prompt, context));

    [Fact]
    public void APromptThatFillsTheContextIsRefused()
    {
        ArgumentException error = Assert.Throws<ArgumentException>(() => MossVlGenerator.ClampToContext(16, 32_768, 32_768));
        Assert.Contains("32768-token context", error.Message);
    }
}
