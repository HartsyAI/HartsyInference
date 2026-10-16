using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Letter-by-letter spelling of words the dictionary does not pronounce (AI, PC, km, OK), checked against the
/// IPA piper_phonemize 1.5.0 produces with the same espeak data. Gated on the espeak data directory like the other
/// phonemizer tests. The negative cases are words espeak reads, not spells, despite a flag-only entry or an
/// all-capitals form.</summary>
public sealed class EspeakLetterSpellingTests
{
    private static EspeakPhonemizer? Build()
    {
        if (!EspeakPhonemizer.TryFindDataDirectory(out string? dataDir) || dataDir is null) return null;
        return EspeakPhonemizer.FromDataDirectory(dataDir, "en-us");
    }

    [Fact]
    public void NonEnglishVoices_DoNotSpellWithAmericanLetterNames()
    {
        // The American letter names must not reach another language's dictionary: "PC" is read by the rules there.
        if (!EspeakPhonemizer.TryFindDataDirectory(out string? dataDir) || dataDir is null) return;
        EspeakPhonemizer spanish = EspeakPhonemizer.FromDataDirectory(dataDir, "es");
        Assert.DoesNotContain("pˌiːsˈiː", spanish.PhonemizeToIpa("PC", "es"));
    }

    [Fact]
    public void BritishVoice_DoesNotSpellWithAmericanLetterNames()
    {
        // The American name of z is "zee"; British English reads the same letter as "zed".
        if (!EspeakPhonemizer.TryFindDataDirectory(out string? dataDir) || dataDir is null) return;
        EspeakPhonemizer british = EspeakPhonemizer.FromDataDirectory(dataDir, "en-gb");
        Assert.DoesNotContain("zˈiː", british.PhonemizeToIpa("NZ", "en-gb"));
    }

    [Theory]
        [InlineData("AI", "ˌeɪˈaɪ")]
        [InlineData("PC", "pˌiːsˈiː")]
        [InlineData("PM", "pˌiːˈɛm")]
        [InlineData("USA", "jˌuːˌɛsˈeɪ")]
        [InlineData("IBM", "ˌaɪbˌiːˈɛm")]
        [InlineData("WWW", "dˌʌbəljˌuːdˌʌbəljˌuːdˈʌbəljˌuː")]
        [InlineData("HTML", "ˌeɪtʃtˌiːˌɛmˈɛl")]
        [InlineData("km", "kˌeɪˈɛm")]
        [InlineData("OK", "ˌoʊkˈeɪ")]
        [InlineData("ok", "ˌoʊkˈeɪ")]
        [InlineData("btw", "bˌiːtˌiːdˈʌbəljˌuː")]
        [InlineData("th", "tˌiːˈeɪtʃ")]
        [InlineData("HR", "ˌeɪtʃˈɑːɹ")]
        [InlineData("mph", "ˌɛmpˌiːˈeɪtʃ")]
        [InlineData("kg", "kˌeɪdʒˈiː")]
        [InlineData("DVD", "dˌiːvˌiːdˈiː")]
        [InlineData("PDF", "pˌiːdˌiːˈɛf")]
        [InlineData("tsp", "tˌiːˌɛspˈiː")]
        [InlineData("Blvd", "bˌiːˌɛlvˌiːdˈiː")]
        [InlineData("CNN", "sˌiːˌɛnˈɛn")]
        [InlineData("MLB", "ˌɛmˌɛlbˈiː")]
        [InlineData("SSD", "ˌɛsˌɛsdˈiː")]
        [InlineData("ASAP", "ˌeɪˌɛsˌeɪpˈiː")]
        [InlineData("KM", "kˌeɪˈɛm")]
        [InlineData("cm", "sˌiːˈɛm")]
        [InlineData("ml", "ˌɛmˈɛl")]
        [InlineData("mg", "ˌɛmdʒˈiː")]
        [InlineData("ft", "ˌɛftˈiː")]
        [InlineData("lb", "ˌɛlbˈiː")]
        [InlineData("MD", "ˌɛmdˈiː")]
        [InlineData("NYPD", "ˌɛnwˌaɪpˌiːdˈiː")]
        [InlineData("GPS", "dʒˌiːpˌiːˈɛs")]
        [InlineData("SMS", "ˌɛsˌɛmˈɛs")]
        [InlineData("VP", "vˌiːpˈiː")]
        [InlineData("TV", "tˌiːvˈiː")]
        [InlineData("NFL", "ˌɛnˌɛfˈɛl")]
        [InlineData("st", "sənt")]
        [InlineData("if", "ɪf")]
        [InlineData("will", "wˈɪl")]
        [InlineData("would", "wˈʊd")]
        [InlineData("never", "nˈɛvɚ")]
        [InlineData("just", "dʒˈʌst")]
    public void SpellsOrReadsLikePiper(string word, string expected)
    {
        EspeakPhonemizer? p = Build();
        if (p is null) return; // gated

        Assert.Equal(expected, p.PhonemizeToIpa(word, "en-us").Replace(" ", string.Empty));
    }
}
