using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Smoke tests for the letter-to-sound rule interpreter against the real v1.50 English rules. The interpreter
/// produces the pre-stress, pre-dictionary phoneme sequence, so output is compared to the consonant/vowel skeleton of
/// espeak's reference (stress marks and dictionary overrides are added by later layers). Gated on
/// <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakTranslatorTests
{
    private static EspeakTranslator? Build()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string dict = Path.Combine(dir, "en_dict");
        string phontab = Path.Combine(dir, "phontab");
        if (!File.Exists(dict) || !File.Exists(phontab)) return null;
        return new EspeakTranslator(
            EspeakDictFile.Load(dict),
            EspeakPhonemeTable.Load(phontab, "en"),
            EspeakLetters.Latin());
    }

    [Fact]
    public void ProducesPlausiblePhonemesForRuleWords()
    {
        EspeakTranslator? tr = Build();
        if (tr is null) return; // gated

        // (word, expected rule-skeleton substring) — espeak full output shown for reference in comments.
        (string Word, string Contains)[] cases =
        [
            ("test", "tEst"),     // espeak: t'Est
            ("strength", "str"),  // espeak: str'ENT
            ("plant", "pl"),      // espeak: pl'a:nt
        ];

        System.Text.StringBuilder dump = new();
        foreach ((string word, string contains) in cases)
        {
            string ph = tr.TranslateWordToMnemonics(word);
            dump.AppendLine($"{word} -> {ph}");
            Assert.False(string.IsNullOrEmpty(ph), $"empty phonemes for '{word}'");
            Assert.Contains(contains, ph, StringComparison.Ordinal);
        }

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "espeak_translator_smoke.txt"), dump.ToString());
    }
}
