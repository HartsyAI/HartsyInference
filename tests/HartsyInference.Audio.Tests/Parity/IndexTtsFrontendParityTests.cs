using System.Text.Json;
using HartsyInference.Audio.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The IndexTTS-1.5/2.0/2.5 text frontend against upstream on the corpus in
/// <c>Fixtures/Frontend/indextts_frontend.json</c>: normalized text, the CJK-split SentencePiece input, every 2.5
/// text stage and the 1.5/2.0 segment cut of the recorded pieces. The wetext FST is replaced by the recorded calls, so
/// these need no model files; tokenizer ids are checked by <see cref="IndexTtsTokenizerParityTests"/>.</summary>
public sealed class IndexTtsFrontendParityTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("v1_5")]
    [InlineData("v2_0")]
    public void SentencePieceText_MatchesUpstream(string version)
    {
        IndexTtsTextFrontend frontend = IndexTtsFrontendFixture.Frontend(version);
        List<string> failures = [];
        int total = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            total++;
            string text = c.GetProperty("text").GetString()!;
            JsonElement v = c.GetProperty(version);
            string? normalized = IndexTtsFrontendFixture.NullableString(v, "normalized");
            string? cjk = IndexTtsFrontendFixture.NullableString(v, "cjk_tokenized");
            if (normalized is not null && frontend.Normalizer.Normalize(text) != normalized)
                failures.Add($"normalize [{text}]: '{frontend.Normalizer.Normalize(text)}' != '{normalized}'");
            if (cjk is not null && frontend.PrepareSentencePiece(text) != cjk)
                failures.Add($"cjk [{text}]: '{frontend.PrepareSentencePiece(text)}' != '{cjk}'");
        }
        Report(version + " text", total, failures);
    }

    [Theory]
    [InlineData("v1_5")]
    [InlineData("v2_0")]
    public void SentencePieceSegments_MatchUpstream(string version)
    {
        List<string> failures = [];
        int total = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            total++;
            JsonElement v = c.GetProperty(version);
            string[] pieces = IndexTtsFrontendFixture.Strings(v.GetProperty("pieces"));
            int[] ids = IndexTtsFrontendFixture.Ints(v.GetProperty("piece_ids"));
            IndexTts2TextSegmenter.Token[] tokens = [.. pieces.Select((p, i) => new IndexTts2TextSegmenter.Token(p, ids[i]))];
            List<IndexTts2TextSegmenter.Token[]> actual = IndexTts2TextSegmenter.Split(tokens, IndexTtsTextFrontend.DefaultMaxTextTokensPerSegment);
            string[][] expected = [.. v.GetProperty("segments").EnumerateArray().Select(IndexTtsFrontendFixture.Strings)];
            if (expected.Length != actual.Count || !expected.Select((s, i) => s.SequenceEqual(actual[i].Select(static t => t.Piece))).All(static b => b))
                failures.Add($"segments [{c.GetProperty("text").GetString()}]");
        }
        Report(version + " segments", total, failures);
    }

    [Fact]
    public void TiktokenTextStages_MatchUpstream()
    {
        IndexTtsTextFrontend frontend = IndexTtsFrontendFixture.Frontend("v2_5");
        List<string> failures = [];
        int total = 0;
        foreach (JsonElement c in IndexTtsFrontendFixture.Cases)
        {
            total++;
            string text = c.GetProperty("text").GetString()!;
            JsonElement v = c.GetProperty("v2_5");
            IndexTtsTextFrontend.TiktokenStages stages = frontend.PrepareTiktoken(text, v.GetProperty("lang").GetString()!);
            Check(failures, text, "cleaned", stages.Cleaned, v.GetProperty("cleaned").GetString());
            Check(failures, text, "normalized", stages.Normalized, IndexTtsFrontendFixture.NullableString(v, "normalized"));
            Check(failures, text, "lowered", stages.Lowered, v.GetProperty("lowered").GetString());
            Check(failures, text, "annotated", stages.Annotated, v.GetProperty("annotated").GetString());
            Check(failures, text, "prepared", stages.Prepared, v.GetProperty("prepared").GetString());
        }
        Report("v2_5 text", total, failures);
    }

    private static void Check(List<string> failures, string text, string stage, string? actual, string? expected)
    {
        if (actual != expected) failures.Add($"{stage} [{text}]: '{actual}' != '{expected}'");
    }

    private void Report(string label, int total, List<string> failures)
    {
        output.WriteLine($"{label}: {total - failures.Count}/{total} match");
        foreach (string f in failures) output.WriteLine(f);
        Assert.True(failures.Count == 0, $"{label}: {failures.Count} mismatches; first: {failures.FirstOrDefault()}");
    }
}
