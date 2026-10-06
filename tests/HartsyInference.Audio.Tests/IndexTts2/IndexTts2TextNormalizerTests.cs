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
    [InlineData("It's a beautiful day, isn't it? What's your name?", "It is a beautiful day, isn't it? What is your name?")]
    [InlineData("I have 3 apples and 12 oranges.", "I have three apples and twelve oranges.")]
    [InlineData("The year is 2025 and the price is $139.99.", "The year is twenty twenty five and the price is one hundred and thirty nine point nine nine dollars.")]
    [InlineData("It is 98.6 degrees, about 37 C, and 2.5% off!", "It is ninety eight point six degrees, about thirty seven C, and two point five percent off!")]
    [InlineData("Dr. Smith and Mr. Jones met Mrs. Brown.", "doctor Smith and Mister Jones met Misses Brown.")]
    [InlineData("We sold 1,234,567 units, up 15%.", "We sold one million two hundred thirty four thousand five hundred and sixty seven units, up fifteen percent.")]
    [InlineData("Version 2.0 of the app has 100 new features.", "Version two point oh of the app has one hundred new features.")]
    [InlineData("The 1st, 2nd, 3rd, and 4th place winners get $50, $100, and $1,000.", "The first, second, third, and fourth place winners get fifty dollars, one hundred dollars, and one thousand dollars.")]
    [InlineData("A 3-year-old, a 20-minute talk, and a 5-star hotel.", "A three-year-old, a twenty-minute talk, and a five-star hotel.")]
    [InlineData("Temperature: -5 degrees; humidity: 80%.", "Temperature, negative five degrees, humidity, eighty percent.")]
    [InlineData("Room 101, flight AA 1234, gate B12.", "Room one hundred and one, flight AA twelve thirty four, gate B twelve.")]
    [InlineData("I'm 25 years old and I've been here since 2010.", "I'm twenty five years old and I've been here since twenty ten.")]
    [InlineData("Pi is about 3.14159.", "Pi is about three point one four one five nine.")]
    [InlineData("The meeting is at 3 pm tomorrow.", "The meeting is at three PM tomorrow.")]
    [InlineData("He ran 5 km in 25 minutes.", "He ran five kilometers in twenty five minutes.")]
    [InlineData("Wait... what?! No way!!!", "Wait… what?! No way!!!")]
    [InlineData("He said: \"hello\" (loudly) — then left.", "He said, 'hello' 'loudly' - then left.")]
    [InlineData("Good morning. Let us begin.", "Good morning. Let us begin.")]
    public void Normalize_MatchesTheReferenceNormalizer(string input, string expected) =>
        Assert.Equal(expected, IndexTts2TextNormalizer.Normalize(input));

    [Theory]
    [InlineData(1010, "ten ten")]
    [InlineData(1100, "eleven hundred")]
    [InlineData(1101, "eleven oh one")]
    [InlineData(1234, "twelve thirty four")]
    [InlineData(1999, "nineteen ninety nine")]
    [InlineData(2000, "two thousand")]
    [InlineData(2005, "two thousand five")]
    [InlineData(2020, "twenty twenty")]
    [InlineData(2100, "twenty one hundred")]
    [InlineData(4567, "four thousand five hundred and sixty seven")]
    [InlineData(10000, "ten thousand")]
    [InlineData(123456, "one hundred twenty three thousand four hundred and fifty six")]
    [InlineData(1000000, "one million")]
    public void Numbers_AreSpokenLikeTheReference(int n, string expected) =>
        Assert.Equal(expected, IndexTts2TextNormalizer.Normalize($"x {n} y")[2..^2]);

    [Fact]
    public void ChineseText_KeepsDigitsButGetsPunctuationMap() =>
        Assert.Equal("你好,今天是2025年10月6日,天气很好.", IndexTts2TextNormalizer.Normalize("你好，今天是2025年10月6日，天气很好。"));

    [Fact]
    public void UseChinese_FollowsTheReferenceRule()
    {
        Assert.True(IndexTts2TextNormalizer.UseChinese("你好 world"));
        Assert.True(IndexTts2TextNormalizer.UseChinese("12345"));
        Assert.False(IndexTts2TextNormalizer.UseChinese("hello world"));
        Assert.True(IndexTts2TextNormalizer.UseChinese("xuan4 is pinyin"));
    }
}
