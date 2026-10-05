using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Whole sentences (from Alice's Adventures in Wonderland) through the en-us <see cref="EspeakPhonemizer"/>,
/// against espeak-ng 1.52's own IPA (fixture from tools/kokoro/espeak_parity_reference.py). Sentences exercise what
/// single words cannot: unstressed function words, words that read differently by position or by the next word,
/// multi-word entries, quotes and hyphens. Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakSentenceParityTests
{
    [Fact]
    public void SentenceMatchRate()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "en_dict"))) return; // gated

        EspeakPhonemizer p = EspeakPhonemizer.FromDataDirectory(dir, "en-us");
        int total = 0, exact = 0;
        System.Text.StringBuilder misses = new();
        foreach (string line in File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Fixtures", "en_sentence_parity.tsv")))
        {
            string[] f = line.Split('\t');
            if (f.Length < 2) continue;
            string got = p.PhonemizeToIpa(f[0], "en-us");
            total++;
            if (got == f[1]) exact++;
            else if (misses.Length < 4000) misses.AppendLine($"{f[0]}\n  want '{f[1]}'\n  have '{got}'");
        }
        double rate = (double)exact / total;
        // 377/400 at writing; the rest are mostly espeak's all-caps abbreviation guesses and noun/verb homographs.
        Assert.True(rate >= 0.9, $"{exact}/{total} = {rate:P1} sentences exact\n{misses}");
    }
}
