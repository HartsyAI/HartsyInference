using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>Diagnostic: which common dictionary words does the lookup find? Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class CommonWordLookupTests
{
    [Fact]
    public void FindsCommonWords()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return;
        EspeakDictFile dict = EspeakDictFile.Load(Path.Combine(dir, "en_dict"));
        EspeakPhonemeTable phon = EspeakPhonemeTable.Load(Path.Combine(dir, "phontab"), "en-us");
        EspeakWordLookup lookup = new(dict, 0x48);

        // These common dictionary words (4+ letters) were all missed before the TransposeAlphabet hash fix (the hash
        // must cover the compressed prefix + leftover original-word tail).
        // All are real en_list (dictionary) entries that were missed before the TransposeAlphabet hash fix. (Words
        // like "people"/"other" are rule-translated by espeak itself, so they are intentionally excluded.)
        string[] words = ["the", "this", "that", "these", "those", "with", "what", "when", "would", "could", "should",
            "there", "their", "about", "which", "than", "then", "them"];
        System.Text.StringBuilder sb = new();
        int found = 0;
        foreach (string w in words)
        {
            bool hit = lookup.Lookup(w, out EspeakLookupResult r);
            if (hit) found++;
            sb.AppendLine($"{w}: found={hit} phon={(hit ? phon.Decode(r.Phonemes) : "-")}");
        }
        File.WriteAllText("/tmp/common_lookup.txt", sb.ToString());
        Assert.True(found == words.Length, $"only {found}/{words.Length} common words found:\n{sb}");
    }
}
