using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>French tokens that Kokoro's clause splitter hands to the switch check one by one: hyphenated words, elisions,
/// and a bare digit. The expected decisions are libespeak-ng 1.52.0.1's (voice <c>fr</c>), read per token: the first four
/// and the bare digit are not switched; <c>e-mail</c> and <c>know-how</c> are read in English. Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class KokoroFrenchTokenTests
{
    private static string? DataDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        return string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab")) ? null : dir;
    }

    [Theory]
    [InlineData("l'été", false)]
    [InlineData("peut-être", false)]
    [InlineData("vingt-et-un", false)]
    [InlineData("3", false)]
    [InlineData("9am", false)]
    [InlineData("e-mail", true)]
    [InlineData("know-how", true)]
    public void Token_SwitchesOnlyWhereEspeakSwitches(string token, bool expected)
    {
        string? dir = DataDir();
        if (dir is null) return; // gated
        Assert.Equal(expected, EspeakPhonemizer.FromDataDirectory(dir, "fr").SwitchesToEnglish(token));
    }

    [Fact]
    public void SentenceWithoutSwitchingTokens_IsUnchanged()
    {
        // The per-token check finds no English token here, so the clause goes to the French voice whole, as before.
        string? dir = DataDir();
        if (dir is null) return; // gated
        EspeakPhonemizer fr = EspeakPhonemizer.FromDataDirectory(dir, "fr");
        KokoroEspeakG2P plain = new(fr);
        KokoroEspeakG2P withSwitch = new(fr, EspeakPhonemizer.FromDataDirectory(dir, "en"), fr.SwitchesToEnglish);
        const string sentence = "Il a vu l'été, peut-être vingt-et-un 3 fois.";
        Assert.Equal(plain.ToIpa(sentence), withSwitch.ToIpa(sentence));
    }

    [Fact]
    public void EmptyEnglishRun_AddsNoSeparator()
    {
        // The English voice reads "-" as nothing, so the run in the middle is empty. The French runs on either side must
        // meet with one space, and an empty run at either end must leave no stray space.
        string? dir = DataDir();
        if (dir is null) return; // gated
        EspeakPhonemizer fr = EspeakPhonemizer.FromDataDirectory(dir, "fr");
        EspeakPhonemizer en = EspeakPhonemizer.FromDataDirectory(dir, "en");
        Assert.Equal(string.Empty, en.PhonemizeTied("-", '^'));

        KokoroEspeakG2P plain = new(fr);
        KokoroEspeakG2P g2p = new(fr, en, w => w == "-");
        Assert.Equal(plain.ToIpa("bonjour") + " " + plain.ToIpa("monde"), g2p.ToIpa("bonjour - monde"));
        Assert.Equal(plain.ToIpa("bonjour"), g2p.ToIpa("- bonjour"));
        Assert.Equal(plain.ToIpa("bonjour"), g2p.ToIpa("bonjour -"));
    }
}
