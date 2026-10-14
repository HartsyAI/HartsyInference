using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Frontends;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>GPT-SoVITS's English frontend against upstream on the corpus in
/// <c>Fixtures/gptsovits_en_frontend_parity.json</c> (from tools/parity/gptsovits_frontend_dump.py, which runs
/// upstream's own code): normalized text, phones and symbol ids per line, inflect number readings, the g2p_en
/// letter-to-sound model, and TextPreprocessor's cut segments and prompt phones. Every entry must match exactly.</summary>
public sealed class GptSoVitsEnglishFrontendParityTests
{
    private static readonly Lazy<JsonDocument> Fixture = new(() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "gptsovits_en_frontend_parity.json"))));

    private readonly ITestOutputHelper _out;

    public GptSoVitsEnglishFrontendParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Lines_MatchUpstreamNormalizationPhonesAndIds()
    {
        StringBuilder misses = new();
        int total = 0, exact = 0;
        foreach (JsonElement line in Fixture.Value.RootElement.GetProperty("lines").EnumerateArray())
        {
            total++;
            string text = line.GetProperty("text").GetString()!;
            string norm = GptSoVitsEnglishG2P.Normalize(text);
            List<string> phones = GptSoVitsFrontend.CleanText(text, "en").Phones;
            int[] ids = GptSoVitsSymbols.ToSequence(phones);
            string wantNorm = line.GetProperty("norm").GetString()!;
            string wantPhones = line.GetProperty("phones").GetString()!;
            int[] wantIds = Ints(line.GetProperty("ids"));
            if (norm == wantNorm && string.Join(' ', phones) == wantPhones && ids.AsSpan().SequenceEqual(wantIds))
            {
                exact++;
                continue;
            }
            misses.AppendLine($"text : {text}");
            if (norm != wantNorm) misses.AppendLine($"  norm want {wantNorm}\n  norm got  {norm}");
            misses.AppendLine($"  want {wantPhones}\n  got  {string.Join(' ', phones)}");
        }
        _out.WriteLine($"{exact}/{total} lines exact");
        Assert.True(total >= 100, $"fixture has only {total} lines");
        Assert.True(exact == total, $"{exact}/{total} lines exact\n{misses}");
    }

    [Fact]
    public void Numbers_NormalizeAsUpstreamInflect()
    {
        StringBuilder misses = new();
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("numbers").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!, want = entry.GetProperty("norm").GetString()!;
            string got = GptSoVitsEnglishNormalizer.Normalize(text);
            if (got != want) misses.AppendLine($"{text}: want '{want}' got '{got}'");
        }
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    public void LetterToSound_MatchesG2pEnPredict()
    {
        StringBuilder misses = new();
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("lts").EnumerateArray())
        {
            string word = entry.GetProperty("word").GetString()!, want = entry.GetProperty("phones").GetString()!;
            string got = string.Join(' ', G2pEnLetterToSound.English.Predict(word));
            if (got != want) misses.AppendLine($"{word}: want '{want}' got '{got}'");
        }
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    public void Segments_MatchUpstreamTextPreprocessor()
    {
        StringBuilder misses = new();
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("segments").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            GptSoVitsCutMethod method = Enum.Parse<GptSoVitsCutMethod>(entry.GetProperty("method").GetString()!, ignoreCase: true);
            List<GptSoVitsTextSegment> got = GptSoVitsTextSegmenter.PrepareEnglish(text, method);
            JsonElement want = entry.GetProperty("segments");
            bool same = got.Count == want.GetArrayLength();
            for (int i = 0; same && i < got.Count; i++)
            {
                same = got[i].NormalizedText == want[i].GetProperty("norm").GetString()
                    && GptSoVitsSymbols.ToSequence(got[i].Phones).AsSpan().SequenceEqual(Ints(want[i].GetProperty("ids")));
            }
            if (same) continue;
            misses.AppendLine($"{method} '{text}'");
            foreach (JsonElement w in want.EnumerateArray()) misses.AppendLine($"  want '{w.GetProperty("norm").GetString()}'");
            foreach (GptSoVitsTextSegment g in got) misses.AppendLine($"  got  '{g.NormalizedText}' {string.Join(' ', g.Phones)}");
        }
        Assert.True(misses.Length == 0, misses.ToString());
    }

    [Fact]
    public void Prompts_MatchUpstreamReferencePhones()
    {
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("prompts").EnumerateArray())
        {
            GptSoVitsTextSegment got = GptSoVitsTextSegmenter.PrepareEnglishPrompt(entry.GetProperty("text").GetString()!);
            Assert.Equal(entry.GetProperty("norm").GetString(), got.NormalizedText);
            Assert.Equal(Ints(entry.GetProperty("ids")), GptSoVitsSymbols.ToSequence(got.Phones));
        }
    }

    /// <summary>The seeded random sweep (<c>gptsovits_frontend_dump.py --stress N --stress-out file.jsonl</c>), gated on
    /// <c>GPTSOVITS_EN_STRESS_FIXTURE</c> naming that file.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void StressLines_MatchUpstream()
    {
        string? path = Environment.GetEnvironmentVariable("GPTSOVITS_EN_STRESS_FIXTURE");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _out.WriteLine("SKIPPED: GPTSOVITS_EN_STRESS_FIXTURE not set");
            return;
        }
        StringBuilder misses = new();
        int total = 0, exact = 0;
        foreach (string json in File.ReadLines(path))
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement line = doc.RootElement;
            total++;
            string text = line.GetProperty("text").GetString()!;
            string norm = GptSoVitsEnglishG2P.Normalize(text);
            string phones = string.Join(' ', GptSoVitsFrontend.CleanText(text, "en").Phones);
            if (norm == line.GetProperty("norm").GetString() && phones == line.GetProperty("phones").GetString())
            {
                exact++;
                continue;
            }
            if (total - exact <= 20)
            {
                misses.AppendLine($"text : {text}\n  norm want {line.GetProperty("norm").GetString()}\n  norm got  {norm}");
                misses.AppendLine($"  want {line.GetProperty("phones").GetString()}\n  got  {phones}");
            }
        }
        _out.WriteLine($"{exact}/{total} stress lines exact");
        Assert.True(exact == total, $"{exact}/{total} stress lines exact\n{misses}");
    }

    private static int[] Ints(JsonElement array)
    {
        int[] values = new int[array.GetArrayLength()];
        int i = 0;
        foreach (JsonElement v in array.EnumerateArray()) values[i++] = v.GetInt32();
        return values;
    }
}
