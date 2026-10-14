using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Chatterbox's text frontend against upstream <c>punc_norm</c> + <c>EnTokenizer</c> + start/stop padding on
/// the corpus in <c>Fixtures/chatterbox_frontend_parity.json</c> (from tools/parity/chatterbox_frontend_dump.py).</summary>
public sealed class ChatterboxFrontendParityTests
{
    private readonly ITestOutputHelper _out;

    public ChatterboxFrontendParityTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData("", ChatterboxTextNormalizer.EmptyTextPrompt)]
    [InlineData(" ", ".")]
    [InlineData("yes", "Yes.")]
    [InlineData("  leading lowercase", "leading lowercase.")]
    [InlineData("Wait... what?", "Wait,  what?")]
    [InlineData("ßharp", "SSharp.")]
    public void Normalize_FollowsPuncNorm(string text, string expected) =>
        Assert.Equal(expected, ChatterboxTextNormalizer.Normalize(text));

    [Fact]
    public void FirstCharacterCapitalization_MatchesPythonOnEveryCodePoint()
    {
        using JsonDocument fixture = OpenFixture();
        Dictionary<int, string> want = new();
        foreach (JsonProperty entry in fixture.RootElement.GetProperty("capitalize").EnumerateObject())
        {
            want[int.Parse(entry.Name, NumberStyles.HexNumber, CultureInfo.InvariantCulture)] = entry.Value.GetString()!;
        }
        int mismatches = 0;
        StringBuilder misses = new();
        for (int cp = 0; cp <= 0x10FFFF; cp++)
        {
            if (!Rune.IsValid(cp)) continue;
            string c = char.ConvertFromUtf32(cp);
            string expected = want.TryGetValue(cp, out string? upper) ? upper : c;
            string got = ChatterboxTextNormalizer.CapitalizeFirst(c + "x")[..^1];
            if (got == expected) continue;
            mismatches++;
            if (misses.Length < 2000) misses.AppendLine($"U+{cp:X4}: want {JsonSerializer.Serialize(expected)} have {JsonSerializer.Serialize(got)}");
        }
        string unicode = fixture.RootElement.GetProperty("unicode").GetString()!;
        Assert.True(mismatches == 0, $"{mismatches} code points differ from Python (Unicode {unicode})\n{misses}");
    }

    [Fact]
    public void NormalizedTextAndTokenIds_MatchUpstreamExactly()
    {
        using JsonDocument fixture = OpenFixture();
        ChatterboxEnTokenizer tokenizer = new();
        int total = 0, matched = 0;
        StringBuilder misses = new();
        foreach (JsonElement line in fixture.RootElement.GetProperty("lines").EnumerateArray())
        {
            string text = line.GetProperty("text").GetString()!;
            string wantText = line.GetProperty("normalized").GetString()!;
            int[] wantIds = line.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            string gotText = ChatterboxTextNormalizer.Normalize(text);
            int[] gotIds = tokenizer.EncodeWithStartStop(gotText);
            total++;
            if (gotText == wantText && gotIds.AsSpan().SequenceEqual(wantIds))
            {
                matched++;
                continue;
            }
            misses.AppendLine($"{JsonSerializer.Serialize(text)}\n want {JsonSerializer.Serialize(wantText)} "
                + $"[{string.Join(',', wantIds)}]\n have {JsonSerializer.Serialize(gotText)} [{string.Join(',', gotIds)}]");
        }
        _out.WriteLine($"chatterbox frontend: {matched}/{total} lines exact");
        Assert.True(total >= 80, $"fixture has only {total} lines");
        Assert.True(matched == total, $"{matched}/{total} lines exact\n{misses}");
    }

    private static JsonDocument OpenFixture() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "chatterbox_frontend_parity.json")));
}
