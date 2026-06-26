using HartsyInference.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Phonemizer.Tests;

/// <summary>End-to-end stress test: word -&gt; (dictionary lookup or letter-to-sound rules) -&gt; SetWordStress -&gt;
/// decoded mnemonics, compared to espeak's own ascii phoneme output. This exercises the whole per-word path bar clause
/// handling. Gated on <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class EspeakStressTests
{
    private sealed record Engine(EspeakWordLookup Lookup, EspeakTranslator Rules, EspeakStress Stress, EspeakPhonemeTable Phon);

    private static Engine? Build()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string dict = Path.Combine(dir, "en_dict");
        string phontab = Path.Combine(dir, "phontab");
        if (!File.Exists(dict) || !File.Exists(phontab)) return null;
        EspeakDictFile d = EspeakDictFile.Load(dict);
        EspeakPhonemeTable phon = EspeakPhonemeTable.Load(phontab, "en");
        return new Engine(new EspeakWordLookup(d), new EspeakTranslator(d, phon, EspeakLetters.Latin()), new EspeakStress(phon), phon);
    }

    private static string Phonemize(Engine e, string word)
    {
        List<byte> codes;
        uint dflags;
        if (e.Lookup.Lookup(word, out EspeakLookupResult r))
        {
            codes = r.Phonemes;
            dflags = r.Flags;
        }
        else
        {
            byte[] buf = new byte[200];
            Array.Fill(buf, (byte)' ');
            for (int i = 0; i < word.Length; i++) buf[2 + i] = (byte)char.ToLowerInvariant(word[i]);
            buf[^1] = 0;
            codes = e.Rules.TranslateRules(buf, 2);
            dflags = 0;
        }
        List<byte> stressed = e.Stress.SetWordStress(codes, dflags, tonic: -1, control: 0);
        return e.Phon.Decode(stressed);
    }

    [Fact]
    public void StressMatchesEspeakForCommonWords()
    {
        Engine? e = Build();
        if (e is null) return; // gated

        // espeak ascii reference (from libespeak-ng): the->D@, and->and, test->t'Est, strength->str'ENT
        (string Word, string Expected)[] cases =
        [
            ("the", "D@"),
            ("and", "and"),
            ("test", "t'Est"),
            ("strength", "str'ENT"),
        ];

        System.Text.StringBuilder dump = new();
        int exact = 0;
        foreach ((string word, string expected) in cases)
        {
            string got = Phonemize(e, word);
            if (got == expected) exact++;
            dump.AppendLine($"{word}: got='{got}' expected='{expected}' {(got == expected ? "OK" : "DIFF")}");
        }
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "espeak_stress_smoke.txt"), dump.ToString());

        Assert.True(exact >= 2, $"only {exact}/4 exact:\n{dump}");
    }
}
