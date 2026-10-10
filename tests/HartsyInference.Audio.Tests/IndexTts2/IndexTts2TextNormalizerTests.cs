using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The English text normalization IndexTTS-2.0 applies before tokenizing. Expected strings are what the
/// reference's own <c>TextNormalizer.normalize</c> (WeTextProcessing) produced for the same inputs, for the cases the
/// port covers.</summary>
public sealed class IndexTts2TextNormalizerTests
{
    [Theory]
    [InlineData("Hello there. This is a real test of index tts 2.0 voice cloning.", "Hello there. This is a real test of index tts two point oh voice cloning.")]
    [InlineData("We sold 1,234,567 units, up 15%.", "We sold one million two hundred thirty four thousand five hundred and sixty seven units, up fifteen percent.")]
    [InlineData("I'm 25 years old and I've been here since 2010.", "I'm twenty five years old and I've been here since twenty ten.")]
    [InlineData("Good morning. Let us begin.", "Good morning. Let us begin.")]
    public void Normalize_MatchesTheReferenceNormalizer(string input, string expected) =>
        Assert.Equal(expected, IndexTts2TextNormalizer.Normalize(input));

    [Theory]
    [InlineData(1010, "ten ten")]
    [InlineData(2020, "twenty twenty")]
    [InlineData(1000000, "one million")]
    public void Numbers_AreSpokenLikeTheReference(int n, string expected) =>
        Assert.Equal(expected, IndexTts2TextNormalizer.Normalize($"x {n} y")[2..^2]);

    [Fact]
    public void UseChinese_FollowsTheReferenceRule()
    {
        Assert.True(IndexTts2TextNormalizer.UseChinese("你好 world"));
        Assert.True(IndexTts2TextNormalizer.UseChinese("12345"));
        Assert.False(IndexTts2TextNormalizer.UseChinese("hello world"));
        Assert.True(IndexTts2TextNormalizer.UseChinese("xuan4 is pinyin"));
    }
}
