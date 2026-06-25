using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>Unit tests for the English (default Latin) letter classification and accent folding ported from
/// espeak-ng. These need no external data file: the bit table is built from the constant letter sets in
/// <c>NewTranslator</c>.</summary>
public sealed class EspeakLettersTests
{
    private static readonly EspeakLetters Letters = EspeakLetters.Latin();

    [Theory]
    [InlineData('a')]
    [InlineData('e')]
    [InlineData('i')]
    [InlineData('o')]
    [InlineData('u')]
    [InlineData('y')] // y is a vowel in the include-y group (LETTERGP_VOWEL2)
    public void VowelsAreVowels(char c) => Assert.True(Letters.IsVowel(c));

    [Theory]
    [InlineData('b')]
    [InlineData('c')]
    [InlineData('z')]
    public void ConsonantsAreNotVowels(char c) => Assert.False(Letters.IsVowel(c));

    [Fact]
    public void ConsonantGroupsMatchEspeakSets()
    {
        // Group 2 (C) is all consonants; group 1 (B) excludes h, r, w.
        Assert.True(Letters.IsLetter('h', EspeakRuleCodes.LetterGpC));
        Assert.False(Letters.IsLetter('h', EspeakRuleCodes.LetterGpB));
        Assert.True(Letters.IsLetter('b', EspeakRuleCodes.LetterGpB));

        // Group 6 (Y) is the front vowels e, i, y only.
        Assert.True(Letters.IsLetter('e', EspeakRuleCodes.LetterGpY));
        Assert.False(Letters.IsLetter('a', EspeakRuleCodes.LetterGpY));
    }

    [Theory]
    [InlineData('é')] // é -> e (vowel)
    [InlineData('è')] // è -> e
    [InlineData('à')] // à -> a
    public void AccentedVowelsFoldToVowels(char c) => Assert.True(Letters.IsVowel(c));
}
