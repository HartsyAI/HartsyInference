using HartsyInference.Audio.Models.ZipVoice;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Cases of <see cref="ZipVoiceEnglishNormalizer"/> the upstream fixture does not reach.</summary>
public sealed class ZipVoiceEnglishNormalizerTests
{
    [Fact]
    public void ANumberPastTheDecillions_KeepsItsDigits()
    {
        string digits = new('7', 40);
        Assert.Equal($" {digits} ", ZipVoiceEnglishNormalizer.Normalize(digits));
    }

    [Fact]
    public void AnAmountThatIsNotANumber_DoesNotFailTheText() =>
        Assert.Equal("$, five", ZipVoiceEnglishNormalizer.Normalize("$,5").Trim());

    [Theory]
    [InlineData("Mr. Smith", "mister. Smith")]
    [InlineData("the St. Louis co", "the saint. Louis company")]
    [InlineData("1,250 people", " twelve fifty  people")]
    [InlineData("3rd of 1/2", " third  of  one half ")]
    public void Rules_ApplyInUpstreamsOrder(string text, string expected) =>
        Assert.Equal(expected, ZipVoiceEnglishNormalizer.Normalize(text));
}
