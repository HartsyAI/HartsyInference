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

    [Fact]
    public void NormalizedTextAndTokenIds_MatchUpstreamExactly()
    {
        using JsonDocument fixture = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "chatterbox_frontend_parity.json")));
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
}
