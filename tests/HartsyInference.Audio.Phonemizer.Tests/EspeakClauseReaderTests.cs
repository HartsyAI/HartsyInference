using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>The clause boundaries and marks of <see cref="EspeakClauseReader"/>, as piper_phonemize reports them for the same text
/// (checked against piper_phonemize 1.5.0): a mark ends a clause only before a space, the end or other punctuation; a period
/// before a lowercase word is an abbreviation dot; an ellipsis and a spaced dash end a clause without or with a semicolon.</summary>
public sealed class EspeakClauseReaderTests
{
    [Theory]
    [InlineData("cat, dog", "cat|,", " dog|")]
    [InlineData("cat; dog", "cat|;", " dog|")]
    [InlineData("cat: dog", "cat|:", " dog|")]
    [InlineData("cat? dog", "cat|?", " dog|")]
    [InlineData("cat! dog", "cat|!", " dog|")]
    [InlineData("cat. Dog", "cat|.", " Dog|")]
    [InlineData("cat. 5 dogs", "cat|.", " 5 dogs|")]
    [InlineData("cat?! dog", "cat|?", " dog|")]
    [InlineData("cat.\" Dog", "cat|.", "\" Dog|")]
    [InlineData("cat… dog", "cat|", " dog|")]
    [InlineData("cat... Dog", "cat|", " Dog|")]
    [InlineData("cat — dog", "cat |;", " dog|")]
    [InlineData("cat— dog", "cat|;", " dog|")]
    [InlineData("cat.", "cat|.")]
    [InlineData("cat,", "cat|,")]
    public void Marks_EndTheirClauses(string text, params string[] expected) =>
        Assert.Equal(expected, Read(text));

    [Theory]
    [InlineData("cat. dog")]
    [InlineData("cat.  dog")]
    [InlineData("cat., dog")]
    [InlineData("cat,dog")]
    [InlineData("cat;dog")]
    [InlineData("cat - dog")]
    [InlineData("cat—dog")]
    [InlineData("cat \"dog\" fox")]
    [InlineData("cat (dog) fox")]
    public void Punctuation_ThatDoesNotEndAClause_LeavesOneClause(string text) =>
        Assert.Single(EspeakClauseReader.Read(text));

    [Fact]
    public void AStandaloneDotBeforeALowercaseWord_IsSpoken() =>
        Assert.Contains("dot", EspeakClauseReader.Read("one . apples")[0].Text);

    [Fact]
    public void ABlankTextHasNoClauses() => Assert.Empty(EspeakClauseReader.Read("   "));

    private static string[] Read(string text) =>
        EspeakClauseReader.Read(text).Select(c => $"{c.Text}|{(c.Mark == '\0' ? "" : c.Mark.ToString())}").ToArray();
}
