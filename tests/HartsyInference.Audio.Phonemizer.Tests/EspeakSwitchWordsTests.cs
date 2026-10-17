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
    /// <summary>The Italian probe: 3,000 random English words (CMU dictionary, seed 7), read by libespeak-ng 1.52.0.1 as
    /// English in <c>woodstock</c>, <c>steve</c>, <c>catherine</c> and <c>teams</c>, and as French in <c>barbecue</c>. This
    /// test fails if a second non-English switch appears, so enabling Italian later cannot misroute one silently. Any
    /// change to the expected sets must come from a fresh libespeak run, not from the port's output.</summary>
    [Fact]
    public void ItalianProbe_NonEnglishSwitchesAreExactlyBarbecue()
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        string[] words = File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "Fixtures", "it_english_probe_words.txt"))
            .Where(w => w.Length > 0).ToArray();
        EspeakPhonemizer it = EspeakPhonemizer.FromDataDirectory(dir, "it");
        EspeakPhonemeTable table = EspeakPhonemeTable.Load(Path.Combine(dir, "phontab"), EspeakVoiceVariant.Resolve(dir, "it").PhonemeTable);
        byte switchCode = (byte)table.CodeForText("_^_");

        string[] english = words.Where(it.SwitchesToEnglish).OrderBy(w => w, StringComparer.Ordinal).ToArray();
        string[] nonEnglish = words.Where(w => it.WordCodes(w).Contains(switchCode) && !it.SwitchesToEnglish(w))
            .OrderBy(w => w, StringComparer.Ordinal).ToArray();

        Assert.Equal(new[] { "catherine", "steve", "teams", "woodstock" }, english);
        Assert.Equal(new[] { "barbecue" }, nonEnglish);
    }
}
