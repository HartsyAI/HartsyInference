using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>Broad parity sweep: runs the <see cref="EspeakPhonemizer"/> over a 400-word fixture and reports the
/// exact-IPA match rate against espeak-ng's own output (fixture generated offline from libespeak-ng v1.50). Gated on
/// <c>ESPEAK_DATA_DIR</c>. The threshold is a regression floor that rises as the remaining espeak refinements land.</summary>
public sealed class EspeakParitySweepTests
{
    [Fact]
    public void SweepMatchRate()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return; // gated
        if (!File.Exists(Path.Combine(dir, "en_dict"))) return;

        string fixture = Path.Combine(AppContext.BaseDirectory, "Fixtures", "en_ipa_parity.tsv");
        if (!File.Exists(fixture)) return;

        EspeakPhonemizer p = EspeakPhonemizer.FromDataDirectory(dir, "en-us");

        int total = 0, exact = 0;
        System.Text.StringBuilder misses = new();
        foreach (string line in File.ReadLines(fixture))
        {
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            string word = line[..tab];
            string want = line[(tab + 1)..];
            string got = p.PhonemizeToIpa(word, "en-us");
            total++;
            if (got == want) exact++;
            else if (misses.Length < 4000) misses.AppendLine($"{word}: got='{got}' want='{want}'");
        }

        double rate = total == 0 ? 0 : (double)exact / total;
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "espeak_parity_sweep.txt"),
            $"exact {exact}/{total} = {rate:P1}\n\n{misses}");

        // Baseline ~65%. Remaining misses are English stress-syllable accuracy, unstressed-vowel reduction
        // (ɒ->ə, a->ɐ), and -s suffix voicing (ps vs pz) — tracked espeak refinements that will raise this floor.
        Assert.True(rate >= 0.62, $"parity {exact}/{total} = {rate:P1} below floor");
    }
}
