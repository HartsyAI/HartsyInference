using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>Broad parity sweep: runs the <see cref="EspeakPhonemizer"/> over a 400-word fixture and reports the
/// exact-IPA match rate against espeak-ng's own output (fixture generated offline from libespeak-ng v1.50). Gated on
/// <c>ESPEAK_DATA_DIR</c>. The threshold is a regression floor that rises as the remaining espeak refinements land.</summary>
public sealed class EspeakParityTests
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

        // ~95% after the phoneme-program VM (data-driven allophones/IPA) + recursive suffix handling + flag-only
        // dictionary fallthrough + prefix stripping (with confirm_prefix + stem stress lock). Remaining misses are a
        // long tail of per-word rule-interpreter vowel choices (ɑː/ə/æ, -ier/-y) and noun/verb stress homographs.
        Assert.True(rate >= 0.94, $"parity {exact}/{total} = {rate:P1} below floor");
    }
}
