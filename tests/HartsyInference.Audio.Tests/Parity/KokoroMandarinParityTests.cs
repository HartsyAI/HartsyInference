using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Frontends;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's Mandarin front-end against misaki's <c>ZHG2P</c> (cn2an 0.5.24, jieba 0.42.1, pypinyin 0.55.0) on
/// the corpus in <c>Fixtures/kokoro_zh_parity.json</c> (from tools/kokoro/mandarin_parity_reference.py): whole
/// sentences, and the jieba cut and pypinyin readings on their own. The sentence test is gated on
/// <c>KOKORO_ZH_DATA_DIR</c>, a folder holding jieba's <c>dict.txt</c> and <c>prob_emit.py</c> and pypinyin's
/// <c>phrases_dict.json</c> and <c>pinyin_dict.json</c>; the syllable table needs no dictionaries.</summary>
public sealed class KokoroMandarinParityTests
{
    private static readonly Lazy<JsonDocument> Fixture = new(() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "kokoro_zh_parity.json"))));

    private readonly ITestOutputHelper _out;

    public KokoroMandarinParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void EveryDictionaryReading_ConvertsAsPypinyinAndMisaki()
    {
        int checkedCount = 0;
        foreach (JsonProperty entry in Fixture.Value.RootElement.GetProperty("syllables").EnumerateObject())
        {
            string tone3 = entry.Value[0].GetString()!, ipa = entry.Value[1].GetString()!;
            Assert.Equal(tone3, PinyinConverter.ToTone3(entry.Name));
            Assert.Equal(ipa, PinyinToIpa.Convert(tone3));
            checkedCount++;
        }
        Assert.True(checkedCount > 1000);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Sentences_AgreeWithMisakiZhG2P()
    {
        string? dir = Environment.GetEnvironmentVariable("KOKORO_ZH_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "dict.txt"))) return; // gated

        KokoroMandarinG2P g2p = KokoroMandarinG2P.FromFiles(Path.Combine(dir, "dict.txt"),
            Path.Combine(dir, "prob_emit.py"), Path.Combine(dir, "phrases_dict.json"), Path.Combine(dir, "pinyin_dict.json"));
        JsonElement root = Fixture.Value.RootElement;
        string[] corpus = Strings(root.GetProperty("corpus"));
        string[] refs = Strings(root.GetProperty("ref"));
        string[] jieba = Strings(root.GetProperty("jieba"));
        string[] pinyin = Strings(root.GetProperty("pinyin"));

        int exact = 0, words = 0, matched = 0, cutExact = 0, pinyinExact = 0;
        StringBuilder misses = new();
        for (int i = 0; i < corpus.Length; i++)
        {
            string got = g2p.ToIpa(corpus[i]);
            string[] want = refs[i].Split(' '), have = got.Split(' ');
            words += want.Length;
            matched += CommonWords(want, have);
            if (got == refs[i]) exact++;
            else if (misses.Length < 4000) misses.AppendLine($"{corpus[i]}\n want '{refs[i]}'\n have '{got}'");

            List<string> cut = HanWords(g2p, corpus[i]);
            if (string.Join(' ', cut) == jieba[i]) cutExact++;
            else if (misses.Length < 4000) misses.AppendLine($"jieba want '{jieba[i]}'\n      have '{string.Join(' ', cut)}'");
            string readings = string.Join(' ', cut.Select(w => string.Join(' ', g2p.Pinyin(w))));
            if (readings == pinyin[i]) pinyinExact++;
            else if (misses.Length < 4000) misses.AppendLine($"pinyin want '{pinyin[i]}'\n       have '{readings}'");
        }
        double rate = (double)matched / words;
        _out.WriteLine($"sentences {exact}/{corpus.Length}, words {matched}/{words} = {rate:P2}, "
            + $"jieba {cutExact}/{corpus.Length}, pinyin {pinyinExact}/{corpus.Length}");
        Assert.True(rate >= 0.99 && cutExact == corpus.Length && pinyinExact == corpus.Length,
            $"{exact}/{corpus.Length} sentences, {rate:P2} of words agree\n{misses}");
    }

    // The jieba words of the sentence's Han runs, after the number and punctuation normalization ToIpa applies.
    private static List<string> HanWords(KokoroMandarinG2P g2p, string text)
    {
        string normalized = ChineseNumberNormalizer.Transform(text);
        List<string> words = [];
        int i = 0;
        while (i < normalized.Length)
        {
            if (normalized[i] < '一' || normalized[i] > '鿿')
            {
                i++;
                continue;
            }
            int start = i;
            while (i < normalized.Length && normalized[i] >= '一' && normalized[i] <= '鿿') i++;
            words.AddRange(g2p.Segment(normalized[start..i]));
        }
        return words;
    }

    private static string[] Strings(JsonElement array) => array.EnumerateArray().Select(e => e.GetString()!).ToArray();

    // Longest common subsequence of words.
    private static int CommonWords(string[] a, string[] b)
    {
        int[,] dp = new int[a.Length + 1, b.Length + 1];
        for (int i = 1; i <= a.Length; i++)
            for (int j = 1; j <= b.Length; j++)
                dp[i, j] = a[i - 1] == b[j - 1] ? dp[i - 1, j - 1] + 1 : Math.Max(dp[i - 1, j], dp[i, j - 1]);
        return dp[a.Length, b.Length];
    }
}
