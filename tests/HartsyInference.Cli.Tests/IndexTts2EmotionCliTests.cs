using HartsyInference.Cli.Dispatch;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The <c>--emotion</c> flag's comma-separated 8-weight parsing (IndexTTS-2's emotion vector).</summary>
public sealed class IndexTts2EmotionCliTests
{
    [Fact]
    public void ParseEmotionVector_ReadsEightInvariantCultureNumbers()
    {
        double[]? v = GenerationDispatch.ParseEmotionVector("0.9, 0,0.2,0 ,0,0,0,0.5");
        Assert.Equal([0.9, 0, 0.2, 0, 0, 0, 0, 0.5], v);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void ParseEmotionVector_UnsetIsNull(string? csv) => Assert.Null(GenerationDispatch.ParseEmotionVector(csv));

    [Theory]
    [InlineData("1,2,3")]
    [InlineData("1,2,3,4,5,6,7,8,9")]
    [InlineData("a,b,c,d,e,f,g,h")]
    public void ParseEmotionVector_RejectsWrongCountOrNonNumbers(string csv)
        => Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseEmotionVector(csv));
}
