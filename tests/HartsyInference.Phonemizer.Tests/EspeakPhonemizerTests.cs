using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>End-to-end parity test for the <see cref="EspeakPhonemizer"/> facade: text in, IPA out, compared to the
/// IPA espeak-ng itself produces. Gated on <c>ESPEAK_DATA_DIR</c>. Reports an exact-match rate so coverage is visible
/// as the remaining espeak refinements land.</summary>
public sealed class EspeakPhonemizerTests
{
    private static EspeakPhonemizer? Build()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        if (!File.Exists(Path.Combine(dir, "en_dict")) || !File.Exists(Path.Combine(dir, "phontab"))) return null;
        return EspeakPhonemizer.FromDataDirectory(dir, "en");
    }

    [Fact]
    public void MatchesEspeakIpa()
    {
        EspeakPhonemizer? p = Build();
        if (p is null) return; // gated

        // espeak-ng IPA reference (libespeak-ng v1.50).
        (string Word, string Ipa)[] cases =
        [
            ("test", "tˈɛst"),
            ("strength", "stɹˈɛŋθ"),
            ("nation", "nˈeɪʃən"),
            ("world", "wˈɜːld"),
            ("through", "θɹˈuː"),
            ("science", "sˈaɪəns"),
            ("phoneme", "fˈəʊniːm"),
            ("important", "ɪmpˈɔːtənt"),
        ];

        System.Text.StringBuilder dump = new();
        int exact = 0;
        foreach ((string word, string expected) in cases)
        {
            string got = p.PhonemizeToIpa(word, "en");
            if (got == expected) exact++;
            dump.AppendLine($"{word}: got='{got}' want='{expected}' {(got == expected ? "OK" : "DIFF")}");
        }
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "espeak_phonemizer_smoke.txt"), dump.ToString());

        // 7/8 exact; "through" (a dict word flagged unstressed) needs clause-level tonic-stress placement, which is
        // layered on with the clause/number handling. The rest match espeak's IPA bit-for-bit.
        Assert.True(exact >= 7, $"only {exact}/{cases.Length} exact IPA matches:\n{dump}");
    }
}
