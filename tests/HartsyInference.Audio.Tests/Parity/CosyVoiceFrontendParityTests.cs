using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>CosyVoice 2's text frontend against the reference Python output (<c>tools/parity/cosyvoice_frontend_dump.py</c>,
/// inflect 7.3.1): digit spell-out, sentence split and the Qwen token ids of every segment. The "plain" fields are the
/// reference path without text-normalization FSTs, which isolates this port from the WeText runtime.</summary>
public sealed class CosyVoiceFrontendParityTests(ITestOutputHelper output)
{
    // CosyVoice 2's own Qwen vocabulary (CosyVoice-BlankEN); its CJK merges differ from the embedded Qwen3 set.
    private const string BlankEnDir = "/mnt/model-storage/Models/audio/tts/FunAudioLLM--CosyVoice2-0.5B/CosyVoice-BlankEN";

    // English token counts are identical under the embedded Qwen vocabulary and CJK splits count characters, so the
    // split itself needs no checkpoint file.
    private static readonly Qwen2Tokenizer Embedded = new();

    private static CosyVoiceTextFrontend PlainFrontend() =>
        new(static s => Embedded.EncodeExact(s, CosyVoiceTextFrontend.SpecialTokens).Length);

    [Fact]
    public void NumberToWords_MatchesInflect()
    {
        using JsonDocument doc = FrontendParityFixture.Load("cosyvoice_frontend");
        foreach (JsonElement n in doc.RootElement.GetProperty("numbers").EnumerateArray())
            Assert.Equal(n.GetProperty("out").GetString(), InflectNumberWords.ToWords(n.GetProperty("in").GetString()!));
    }

    [Fact]
    public void PlainPath_SegmentsAndNormalizedText_MatchReference()
    {
        using JsonDocument doc = FrontendParityFixture.Load("cosyvoice_frontend");
        CosyVoiceTextFrontend frontend = PlainFrontend();
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.Equal(c.GetProperty("plain_normalized").GetString(), frontend.Normalize(text));
            Assert.Equal(FrontendParityFixture.Strings(c.GetProperty("plain_segments")), frontend.Segments(text));
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void SegmentTokenIds_MatchReferenceTokenizer()
    {
        string vocab = Path.Combine(BlankEnDir, "vocab.json");
        string merges = Path.Combine(BlankEnDir, "merges.txt");
        if (!RealWeightGate.Require(output.WriteLine, vocab, merges)) return;

        Qwen2Tokenizer tokenizer = new(vocab, merges);
        using JsonDocument doc = FrontendParityFixture.Load("cosyvoice_frontend");
        List<string> mismatches = [];
        int compared = 0;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            List<string> segments = FrontendParityFixture.Strings(c.GetProperty("plain_segments"));
            List<int[]> expected = FrontendParityFixture.IdLists(c.GetProperty("plain_segment_ids"));
            for (int i = 0; i < segments.Count; i++)
            {
                int[] actual = tokenizer.EncodeExact(segments[i], CosyVoiceTextFrontend.SpecialTokens);
                compared++;
                if (!expected[i].AsSpan().SequenceEqual(actual))
                    mismatches.Add($"'{segments[i]}'\n  ref: {string.Join(' ', expected[i])}\n  got: {string.Join(' ', actual)}");
            }
        }
        output.WriteLine($"compared {compared} segments");
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }
}
