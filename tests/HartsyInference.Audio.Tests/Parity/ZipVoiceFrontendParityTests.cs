using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Models.ZipVoice;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>ZipVoice's text front end against upstream on the corpus in <c>Fixtures/zipvoice_frontend_parity.json</c> (from
/// tools/parity/zipvoice_frontend_dump.py, which runs upstream's own EmiliaTokenizer, EnglishTextNormalizer, add_punctuation and
/// chunk_tokens_punctuation): sentence punctuation, number and abbreviation normalization, the token chunker, and the clause
/// structure of piper_phonemize's output from a word lexicon. The token symbols from the real espeak data need that data.</summary>
public sealed class ZipVoiceFrontendParityTests(ITestOutputHelper output)
{
    [Fact]
    public void AddPunctuation_MatchesUpstream()
    {
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("add_punctuation").EnumerateArray())
        {
            string input = entry.GetProperty("in").GetString()!;
            JsonElement expected = entry.GetProperty("out");
            if (expected.ValueKind == JsonValueKind.Null) Assert.Throws<ArgumentException>(() => ZipVoiceTokenizer.AddPunctuation(input));
            else Assert.Equal(expected.GetString(), ZipVoiceTokenizer.AddPunctuation(input));
        }
    }

    [Fact]
    public void Normalizer_MatchesUpstream()
    {
        StringBuilder misses = new();
        int checkedLines = 0;
        foreach (JsonElement line in ZipVoiceFixture.Root.GetProperty("lines").EnumerateArray())
        {
            if (!line.TryGetProperty("norm", out JsonElement norm) || norm.ValueKind == JsonValueKind.Null) continue;
            checkedLines++;
            string segment = line.GetProperty("segments")[0][0].GetString()!;
            string got = ZipVoiceEnglishNormalizer.Normalize(segment);
            if (got != norm.GetString()) misses.AppendLine($"{segment}\n  want '{norm.GetString()}'\n  got  '{got}'");
        }
        Assert.True(checkedLines >= 100, $"fixture has only {checkedLines} normalizer lines");
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    public void Chunker_MatchesUpstream()
    {
        JsonElement lines = ZipVoiceFixture.Root.GetProperty("lines");
        StringBuilder misses = new();
        int total = 0;
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("chunks").EnumerateArray())
        {
            total++;
            List<string> tokens = Symbols(lines[entry.GetProperty("line").GetInt32()].GetProperty("tokens").GetString()!);
            int budget = entry.GetProperty("max").GetInt32();
            string[] want = entry.GetProperty("chunks").EnumerateArray().Select(e => e.GetString()!).ToArray();
            string[] got = ZipVoiceTextChunker.Chunk(tokens, budget).Select(c => string.Concat(c)).ToArray();
            if (!got.SequenceEqual(want)) misses.AppendLine($"line {entry.GetProperty("line").GetInt32()} budget {budget}: want {want.Length} chunks, got {got.Length}");
        }
        Assert.True(total >= 100, $"fixture has only {total} chunk cases");
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    public void TargetFrames_MatchUpstreamFloat32Arithmetic()
    {
        int total = 0;
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("durations").EnumerateArray())
        {
            total++;
            int got = ZipVoiceModel.PredictTargetFrames(entry.GetProperty("frames").GetInt32(), entry.GetProperty("prompt_tokens").GetInt32(),
                entry.GetProperty("tokens").GetInt32(), (float)entry.GetProperty("speed").GetDouble());
            Assert.Equal(entry.GetProperty("target").GetInt32(), got);
        }
        Assert.True(total >= 400, $"fixture has only {total} duration cases");
    }

    [Fact]
    public void Clauses_MatchPiperPhonemizeFromALexicon()
    {
        Dictionary<string, string> lexicon = ZipVoiceFixture.Root.GetProperty("lexicon").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetString()!);
        ZipVoiceFixture.LexiconPhonemizer phonemizer = new(lexicon);
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        StringBuilder misses = new();
        int total = 0;
        foreach (JsonElement entry in ZipVoiceFixture.Root.GetProperty("clauses").EnumerateArray())
        {
            total++;
            string text = entry.GetProperty("text").GetString()!;
            string want = entry.GetProperty("tokens").GetString()!;
            string got = string.Concat(tokenizer.Tokenize(text, phonemizer));
            if (got != want) misses.AppendLine($"{text}\n  want '{want}'\n  got  '{got}'");
        }
        Assert.True(total >= 100, $"fixture has only {total} clause cases");
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Tokens_AgreeWithUpstreamOnEspeakData()
    {
        string phontab = EspeakPhonemizer.TryFindDataDirectory(out string? dataDir) ? Path.Combine(dataDir, "phontab") : "espeak-ng-data";
        if (!RealWeightGate.Require(output.WriteLine, phontab)) return;

        EspeakPhonemizer phonemizer = EspeakPhonemizer.FromDataDirectory(dataDir!, ZipVoiceTokenizer.EspeakLanguage);
        ZipVoiceTokenizer tokenizer = ZipVoiceFixture.PadOnlyTokenizer();
        StringBuilder misses = new();
        int lines = 0, exact = 0, words = 0, matched = 0;
        foreach (JsonElement line in ZipVoiceFixture.Root.GetProperty("lines").EnumerateArray())
        {
            if (!line.TryGetProperty("tokens", out JsonElement want)) continue;
            lines++;
            string punct = line.GetProperty("punct").GetString()!;
            string got = string.Concat(tokenizer.Tokenize(punct, phonemizer));
            (int total, int agreed) = WordAgreement(want.GetString()!, got);
            words += total;
            matched += agreed;
            if (got == want.GetString()) exact++;
            else misses.AppendLine($"{punct}\n  want {want.GetString()}\n  got  {got}");
        }
        double rate = (double)matched / words;
        output.WriteLine($"{exact}/{lines} lines exact, {matched}/{words} = {rate:P1} of words agree\n{misses}");
        Assert.True(lines >= 130, $"fixture has only {lines} lines");
        // The cached espeak data and port read some words differently from piper's build (acronyms, symbols, a few
        // vowels); a wrong voice, as the British one, falls to about 58%.
        Assert.True(rate >= 0.92, $"{matched}/{words} = {rate:P1} of words agree\n{misses}");
    }

    private static (int Total, int Agreed) WordAgreement(string want, string got)
    {
        char[] separators = [' ', '.', ',', ';', ':', '!', '?'];
        Dictionary<string, int> remaining = new();
        foreach (string word in got.Split(separators, StringSplitOptions.RemoveEmptyEntries)) remaining[word] = remaining.GetValueOrDefault(word) + 1;
        int total = 0, agreed = 0;
        foreach (string word in want.Split(separators, StringSplitOptions.RemoveEmptyEntries))
        {
            total++;
            if (remaining.GetValueOrDefault(word) == 0) continue;
            remaining[word]--;
            agreed++;
        }
        return (total, agreed);
    }

    private static List<string> Symbols(string text)
    {
        List<string> symbols = new();
        foreach (Rune rune in text.EnumerateRunes()) symbols.Add(rune.ToString());
        return symbols;
    }
}
