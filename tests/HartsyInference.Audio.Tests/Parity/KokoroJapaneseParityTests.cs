using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.MeCab;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's Japanese front-end against misaki's <c>JAG2P</c> (cutlet mode, fugashi over UniDic 3.1.0) on the
/// sentences of <c>Fixtures/kokoro_japanese_parity.json</c>, written by
/// <c>tools/kokoro/japanese_parity_reference.py</c>: the MeCab tokens (surface, pron, kana, char_type, unknown flag)
/// and the final phoneme strings. Gated on <c>KOKORO_JA_DICT_DIR</c> naming a directory with the UniDic
/// <c>sys.dic</c>, <c>unk.dic</c>, <c>matrix.bin</c>, <c>char.bin</c> and misaki's <c>ja_words.txt</c>.</summary>
public sealed class KokoroJapaneseParityTests
{
    private readonly ITestOutputHelper _out;

    public KokoroJapaneseParityTests(ITestOutputHelper output) => _out = output;

    private static string? DictionaryDirectory()
    {
        string? dir = Environment.GetEnvironmentVariable("KOKORO_JA_DICT_DIR");
        return string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "sys.dic")) ? null : dir;
    }

    private static List<(string Text, string Ipa, string[] Tokens)> Entries()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "kokoro_japanese_parity.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        List<(string, string, string[])> entries = [];
        foreach (JsonElement e in doc.RootElement.GetProperty("entries").EnumerateArray())
        {
            entries.Add((e.GetProperty("text").GetString()!, e.GetProperty("ipa").GetString()!,
                e.GetProperty("tokens").EnumerateArray().Select(static t => t.GetString()!).ToArray()));
        }
        return entries;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Tokenizer_MatchesFugashi()
    {
        string? dir = DictionaryDirectory();
        if (dir is null) return;
        using MeCabTokenizer tagger = MeCabTokenizer.Open(dir);
        int exact = 0, total = 0;
        foreach ((string text, _, string[] expected) in Entries())
        {
            string[] actual = tagger.Tokenize(KokoroJapaneseG2P.Normalize(text))
                .Select(static t => string.Join('\t', t.Surface, t.Field(9) ?? "", t.Field(20) ?? "",
                    t.CharType.ToString(System.Globalization.CultureInfo.InvariantCulture), t.IsUnknown ? "1" : "0"))
                .ToArray();
            total++;
            if (expected.SequenceEqual(actual)) exact++;
            else _out.WriteLine($"TOKENS {text}\n  ref: {string.Join(" | ", expected)}\n  got: {string.Join(" | ", actual)}");
        }
        _out.WriteLine($"tokenizer: {exact}/{total} sentences identical");
        Assert.Equal(total, exact);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ToIpa_MatchesMisaki()
    {
        string? dir = DictionaryDirectory();
        if (dir is null) return;
        using KokoroJapaneseG2P g2p = new(dir);
        int exact = 0, total = 0;
        long chars = 0, charErrors = 0;
        foreach ((string text, string expected, _) in Entries())
        {
            string actual = g2p.ToIpa(text);
            total++;
            chars += expected.Length;
            charErrors += Levenshtein(expected, actual);
            if (expected == actual) exact++;
            else _out.WriteLine($"IPA {text}\n  ref: {expected}\n  got: {actual}");
        }
        _out.WriteLine($"phonemes: {exact}/{total} sentences identical, character agreement "
            + $"{1 - (double)charErrors / chars:P3}");
        Assert.Equal(total, exact);
    }

    private static int Levenshtein(string a, string b)
    {
        int[] row = Enumerable.Range(0, b.Length + 1).ToArray();
        for (int i = 1; i <= a.Length; i++)
        {
            int diagonal = row[0];
            row[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int above = row[j];
                row[j] = Math.Min(Math.Min(row[j] + 1, row[j - 1] + 1), diagonal + (a[i - 1] == b[j - 1] ? 0 : 1));
                diagonal = above;
            }
        }
        return row[b.Length];
    }
}
