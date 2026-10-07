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

    [Theory]
    // A decimal point stays in its number, a thousands separator before three digits groups it, and any other
    // separator splits the digits; espeak-ng 1.52's own IPA.
    [InlineData("en-us", "it costs 3.5 dollars", "ɪt kˈɔsts θɹˈiː pɔɪnt fˈaɪv dˈɑːlɚz")]
    [InlineData("en-us", "1,234 people", "wˈʌn θˈaʊzənd tˈuːhˈʌndɹɪd θˈɜːɾi fˈɔːɹ pˈiːpəl")]
    [InlineData("en-us", "12,345,678", "twˈɛlv mˈɪliən θɹˈiːhˈʌndɹɪd fˈɔːɹɾi fˈaɪv θˈaʊzənd sˈɪkshˈʌndɹɪd sˈɛvənti ˈeɪt")]
    [InlineData("en-us", "1,23 and 3.14159", "wˈʌn twˈɛnti θɹˈiː ænd θɹˈiː pɔɪnt wˈʌn fˈɔːɹ wˈʌn fˈaɪv nˈaɪn")]
    [InlineData("es", "cuesta 3,5 euros", "kwˈesta tɾˈes koma θˈinko ˈeʊɾos")]
    [InlineData("es", "1.234 personas", "mˈil dosθjˈentos tɾˌeɪntaikwˈatɾo peɾsˈonas")]
    public void NumberSeparators_ReadAsEspeak(string language, string text, string espeak)
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab"))) return; // gated
        Assert.Equal(espeak, EspeakPhonemizer.FromDataDirectory(dir, language).PhonemizeToIpa(text, language));
    }
}
