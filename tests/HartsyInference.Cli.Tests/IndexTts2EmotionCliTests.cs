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
        Assert.NotNull(v);
        Assert.Equal(new[] { 0.9, 0, 0.2, 0, 0, 0, 0, 0.5 }, v);
    }

    [Theory]
    [InlineData("1,2,3")]
    [InlineData("a,b,c,d,e,f,g,h")]
    public void ParseEmotionVector_RejectsWrongCountOrNonNumbers(string csv)
        => Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseEmotionVector(csv));
}
