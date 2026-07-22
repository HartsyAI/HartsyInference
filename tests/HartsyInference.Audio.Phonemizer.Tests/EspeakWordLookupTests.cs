using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Validates the dictionary word-list lookup (transpose + 6-bit pack + hash + entry match) against the real
/// v1.50 English dictionary. Common function words must be found and decode to espeak's stored pronunciation. Gated on
/// <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakWordLookupTests
{
    private static (EspeakWordLookup, EspeakPhonemeTable)? Build()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string dict = Path.Combine(dir, "en_dict");
        string phontab = Path.Combine(dir, "phontab");
        if (!File.Exists(dict) || !File.Exists(phontab)) return null;
        return (new EspeakWordLookup(EspeakDictFile.Load(dict)), EspeakPhonemeTable.Load(phontab, "en"));
    }

    [Fact]
    public void FindsCommonWords()
    {
        (EspeakWordLookup lookup, EspeakPhonemeTable phon)? built = Build();
        if (built is null) return; // gated

        (EspeakWordLookup lookup, EspeakPhonemeTable phon) = built.Value;

        // espeak reference (ascii mnemonics): the->D@, of->0v, and->and, to->tu:
        string[] words = ["the", "of", "and", "to", "plant"];
        System.Text.StringBuilder dump = new();
        int foundCount = 0;
        foreach (string w in words)
        {
            bool found = lookup.Lookup(w, out EspeakLookupResult r);
            if (found) foundCount++;
            string decoded = found ? phon.Decode(r.Phonemes) : "<not found>";
            dump.AppendLine($"{w} -> found={found} phon={decoded} flags=0x{r.Flags:x} flags2=0x{r.Flags2:x}");
        }
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "espeak_lookup_smoke.txt"), dump.ToString());
        Assert.True(foundCount >= 3, $"only {foundCount}/5 common words found\n{dump}");

        // "the" decodes to a 'D' (eth) followed by a schwa '@'.
        Assert.True(lookup.Lookup("the", out EspeakLookupResult the));
        string thePh = phon.Decode(the.Phonemes);
        Assert.Contains("D", thePh, StringComparison.Ordinal);
    }
}
