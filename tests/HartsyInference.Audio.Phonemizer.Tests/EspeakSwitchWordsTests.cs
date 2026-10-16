using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Which words espeak-ng reads in English inside another language, checked word by word against libespeak-ng
/// 1.52.0.1 (voice <c>fr</c> and <c>it</c>). The switch comes from the dictionary entry or the rules, with prefix and
/// suffix stripping, so a word's plural or stem can switch even when the whole word does not appear in a dictionary.
/// Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakSwitchWordsTests
{
    private static string? DataDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        return string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab")) ? null : dir;
    }

    [Theory]
    // espeak-ng 1.52 reads each of these in English within French text: plural or stem switches, and one prefix case.
    [InlineData("strongholds")]
    [InlineData("puddings")]
    [InlineData("postmark")]
    [InlineData("mccleave")]
    [InlineData("dived")]
    public void French_SwitchesWordsEspeakSwitches(string word)
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        Assert.True(EspeakPhonemizer.FromDataDirectory(dir, "fr").SwitchesToEnglish(word), $"'{word}' should switch to English");
    }

    [Theory]
    // espeak-ng 1.52 reads 'teams' in English within Italian text by stripping the plural and switching the stem 'team'.
    [InlineData("teams", true)]
    [InlineData("team", true)]
    // espeak-ng reads 'barbecue' in French within Italian, not English: the dictionary entry carries more than the switch.
    [InlineData("barbecue", false)]
    public void Italian_SwitchesOnlyToEnglish(string word, bool expected)
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        Assert.Equal(expected, EspeakPhonemizer.FromDataDirectory(dir, "it").SwitchesToEnglish(word));
    }
}
