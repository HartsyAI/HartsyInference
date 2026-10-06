using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Phonemizer.Espeak;
using Xunit;

namespace HartsyInference.Audio.Phonemizer.Tests;

/// <summary>The espeak languages Kokoro speaks, read through <see cref="KokoroEspeakG2P"/> and diffed word by word
/// against misaki's <c>EspeakG2P</c> on the same sentences (UDHR text plus numbers, symbols and ordinals; fixture from
/// tools/kokoro/espeak_parity_reference.py over espeak-ng 1.52). Gated on <c>ESPEAK_DATA_DIR</c>, which must hold that
/// data: older espeak-ng data reads some words differently.</summary>
public sealed class KokoroEspeakParityTests
{
    [Theory]
    [InlineData("es")]
    [InlineData("fr-fr")]
    [InlineData("it")]
    [InlineData("pt-br")]
    [InlineData("hi")]
    public void WordAgreement_WithMisakiEspeakG2P(string language)
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab"))) return; // gated

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "kokoro_espeak_parity.json")));
        string[] corpus = Strings(doc.RootElement.GetProperty("corpus").GetProperty(language));
        string[] refs = Strings(doc.RootElement.GetProperty("ref").GetProperty(language));
        KokoroEspeakG2P g2p = new(EspeakPhonemizer.FromDataDirectory(dir, language));

        int words = 0, matched = 0;
        System.Text.StringBuilder misses = new();
        for (int i = 0; i < corpus.Length; i++)
        {
            string got = g2p.ToIpa(corpus[i]);
            string[] want = refs[i].Split(' '), have = got.Split(' ');
            words += want.Length;
            matched += CommonWords(want, have);
            if (got != refs[i] && misses.Length < 4000) misses.AppendLine($"want '{refs[i]}'\nhave '{got}'");
        }
        double rate = (double)matched / words;
        // At writing: es 99.9%, fr-fr 99.6%, it 100%, pt-br 100%, hi 99.5%.
        Assert.True(rate >= 0.99, $"{language}: {matched}/{words} = {rate:P1} of words agree\n{misses}");
    }

    [Theory]
    // Every hyphen joins its own pair of words, however many the clause holds.
    [InlineData("fr-fr", "c'est-à-dire lundi", "sɛtadˈiʁ lœ̃dˈi")]
    [InlineData("es", "hispano-franco-italiano", "ispˈanofɾˈankoˌitaljˈano")]
    public void Hyphens_JoinTheirOwnWords(string language, string text, string misaki)
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab"))) return; // gated
        Assert.Equal(misaki, new KokoroEspeakG2P(EspeakPhonemizer.FromDataDirectory(dir, language)).ToIpa(text));
    }

    [Fact]
    // Kokoro keeps one phonemizer per language and every request shares it, so concurrent reads must match serial ones.
    public void SharedPhonemizer_IsSafeAcrossThreads()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "phontab"))) return; // gated

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "kokoro_espeak_parity.json")));
        string[] corpus = Strings(doc.RootElement.GetProperty("corpus").GetProperty("fr-fr"));
        KokoroEspeakG2P g2p = new(EspeakPhonemizer.FromDataDirectory(dir, "fr-fr"));
        string[] serial = corpus.Select(g2p.ToIpa).ToArray();
        string[] parallel = new string[corpus.Length * 8];
        Parallel.For(0, parallel.Length, i => parallel[i] = g2p.ToIpa(corpus[i % corpus.Length]));
        for (int i = 0; i < parallel.Length; i++) Assert.Equal(serial[i % corpus.Length], parallel[i]);
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
