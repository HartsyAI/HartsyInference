using System.Text.Json;
using HartsyInference.Audio.Pipelines;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The IndexTTS frontend end to end against upstream on <c>Fixtures/Frontend/indextts_frontend.json</c>:
/// SentencePiece pieces, segments and GPT ids for 1.5 and 2.0, and the tiktoken split, per-segment rows (language tag
/// and stop pad included) and GPT ids for 2.5. Needs the released tokenizer files under
/// <c>HARTSY_INDEXTTS_MODELS_ROOT</c> (default the model RAID); the wetext FST is the recorded lookup.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTtsTokenizerParityTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("v1_5", "IndexTeam--IndexTTS-1.5")]
    [InlineData("v2_0", "IndexTeam--IndexTTS-2")]
    public void SentencePiece_PiecesSegmentsAndIds_MatchUpstream(string version, string repo)
    {
        string bpe = Path.Combine(IndexTtsFrontendFixture.ModelsRoot, repo, "bpe.model");
        if (!RealWeightGate.Require(output.WriteLine, bpe)) return;
        using IndexTtsTokenizer tokenizer = new(bpe);
        IndexTtsTextFrontend frontend = IndexTtsFrontendFixture.Frontend(version);
        List<string> failures = [];
        int total = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            total++;
            string text = c.GetProperty("text").GetString()!;
            JsonElement v = c.GetProperty(version);
            string? prepared = frontend.PrepareSentencePiece(text);
            string[] pieces = prepared is null ? [] : [.. tokenizer.EncodeToPieces(prepared).Select(static p => p.Piece)];
            if (!pieces.SequenceEqual(IndexTtsFrontendFixture.Strings(v.GetProperty("pieces"))))
            {
                failures.Add($"pieces [{text}]: {string.Join('|', pieces)}");
                continue;
            }
            List<int[]> ids = frontend.SegmentSentencePiece(text, tokenizer);
            int[][] expected = [.. v.GetProperty("gpt_ids").EnumerateArray().Select(IndexTtsFrontendFixture.Ints)];
            if (!SameRows(expected, ids)) failures.Add($"ids [{text}]");
        }
        Report(version + " tokens", total, failures);
    }

    [Fact]
    public void Tiktoken_SegmentsRowsAndIds_MatchUpstream()
    {
        string path = Path.Combine(IndexTtsFrontendFixture.ModelsRoot, "IndexTeam--IndexTTS-2.5", "multilingual_zh_ja_yue_char_del.tiktoken");
        if (!RealWeightGate.Require(output.WriteLine, path)) return;
        IndexTts2TiktokenTokenizer tokenizer = new(path);
        IndexTtsTextFrontend frontend = IndexTtsFrontendFixture.Frontend("v2_5");
        int capacity = IndexTtsFrontendFixture.Root.GetProperty("v2_5_text_pos_capacity").GetInt32();
        List<string> failures = [];
        int total = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            total++;
            string text = c.GetProperty("text").GetString()!;
            JsonElement v = c.GetProperty("v2_5");
            string lang = v.GetProperty("lang").GetString()!;
            string prefix = IndexTtsTextFrontend.LanguagePrefix(lang);
            List<string> segments = IndexTtsTextFrontend.SplitByTokens(v.GetProperty("prepared").GetString()!,
                IndexTtsTextFrontend.DefaultMaxTextTokensPerSegment, capacity, prefix, s => tokenizer.Encode(s).Length);
            if (!segments.SequenceEqual(IndexTtsFrontendFixture.Strings(v.GetProperty("segments"))))
            {
                failures.Add($"segments [{text}]: {string.Join(" / ", segments)}");
                continue;
            }
            List<int[]> rows = [.. segments.Select(s => IndexTtsTextFrontend.EncodeTiktokenSegment(s, prefix, tokenizer.Encode))];
            if (!SameRows([.. v.GetProperty("segment_ids").EnumerateArray().Select(IndexTtsFrontendFixture.Ints)], rows))
            {
                failures.Add($"rows [{text}]: {string.Join(" / ", rows.Select(static r => string.Join(',', r)))}");
                continue;
            }
            List<int[]> ids = frontend.SegmentTiktoken(text, lang, tokenizer.Encode, capacity);
            if (!SameRows([.. v.GetProperty("gpt_ids").EnumerateArray().Select(IndexTtsFrontendFixture.Ints)], ids)) failures.Add($"ids [{text}]");
        }
        Report("v2_5 tokens", total, failures);
    }

    private static bool SameRows(int[][] expected, List<int[]> actual) =>
        expected.Length == actual.Count && expected.Select((row, i) => row.SequenceEqual(actual[i])).All(static b => b);

    private void Report(string label, int total, List<string> failures)
    {
        output.WriteLine($"{label}: {total - failures.Count}/{total} match");
        foreach (string f in failures) output.WriteLine(f);
        Assert.True(failures.Count == 0, $"{label}: {failures.Count} mismatches; first: {failures.FirstOrDefault()}");
    }
}
