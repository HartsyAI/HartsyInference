using System.Text.Json;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Frontends.WeText;
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
            Assert.Equal(n.GetProperty("out").GetString(), InflectNumberWords.NumberToWords(n.GetProperty("in").GetString()!));
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

    /// <summary>The full reference path with wetext 0.0.4 normalization. The FST normalizer is injected as a lookup of
    /// the outputs wetext produced for the same inputs, so everything around it (spell-out, split, Chinese rewrites)
    /// is checked exactly; the WeText runtime has its own parity tests.</summary>
    [Fact]
    public void NormalizedPath_WithReferenceFstOutputs_MatchesReference()
    {
        using JsonDocument doc = FrontendParityFixture.Load("cosyvoice_frontend");
        Dictionary<string, string> tnRaw = [];
        foreach (JsonProperty p in doc.RootElement.GetProperty("tn_raw").EnumerateObject()) tnRaw[p.Name] = p.Value.GetString()!;
        string Lookup(string s) => tnRaw.TryGetValue(s, out string? v) ? v : throw new KeyNotFoundException($"No reference FST output for '{s}'.");
        CosyVoiceTextFrontend frontend = new(static s => Embedded.EncodeExact(s, CosyVoiceTextFrontend.SpecialTokens).Length, Lookup, Lookup);
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.Equal(c.GetProperty("tn_normalized").GetString(), frontend.Normalize(text));
            Assert.Equal(FrontendParityFixture.Strings(c.GetProperty("tn_segments")), frontend.Segments(text));
        }
    }

    /// <summary>The full reference path through the C# WeText runtime (the engine's real configuration): normalized text
    /// and segments must equal the reference built with wetext 0.0.4.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void NormalizedPath_WithWeTextRuntime_MatchesReference()
    {
        string fstDir = Environment.GetEnvironmentVariable("WETEXT_FST_DIR")
            ?? "/mnt/model-storage/scratch/tts-ref/fe-venv/lib/python3.12/site-packages/wetext/fsts";
        if (!RealWeightGate.Require(output.WriteLine, [.. WeTextNormalizer.RequiredFiles().Select(f => Path.Combine(fstDir, f))])) return;

        WeTextNormalizer weText = WeTextNormalizer.Load(fstDir);
        string Tn(string s) => weText.TryNormalize(s, out string? n) ? n! : s;
        CosyVoiceTextFrontend frontend = new(static s => Embedded.EncodeExact(s, CosyVoiceTextFrontend.SpecialTokens).Length, Tn, Tn);
        using JsonDocument doc = FrontendParityFixture.Load("cosyvoice_frontend");
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.Equal(c.GetProperty("tn_normalized").GetString(), frontend.Normalize(text));
            Assert.Equal(FrontendParityFixture.Strings(c.GetProperty("tn_segments")), frontend.Segments(text));
        }
    }

    /// <summary>The added-token ids must lie outside the BPE vocabulary, so a special literal and an ordinary merge can
    /// never share an id.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void SpecialTokenIds_AreOutsideTheBpeVocabulary()
    {
        string vocab = Path.Combine(BlankEnDir, "vocab.json");
        if (!RealWeightGate.Require(output.WriteLine, vocab)) return;

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(vocab));
        HashSet<int> bpeIds = [.. doc.RootElement.EnumerateObject().Select(static p => p.Value.GetInt32())];
        foreach ((string literal, int id) in CosyVoiceTextFrontend.SpecialTokens)
            Assert.False(bpeIds.Contains(id), $"{literal} ({id}) collides with a vocab.json entry");
        Assert.Equal(CosyVoiceTextFrontend.SpecialTokens.Count, CosyVoiceTextFrontend.SpecialTokens.Values.Distinct().Count());
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
