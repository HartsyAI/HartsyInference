using System.Text.Json;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The token-level segment splitter against the reference <c>TextTokenizer.split_segments</c>: the fixture
/// holds, for a dozen prompts, the real SentencePiece pieces the reference produced and the segments it cut them into
/// (at the default 120-token limit, with <c>quick_streaming_tokens=20</c>, and at a 24-token limit that forces
/// comma/hyphen/length splitting). The splitter needs only the pieces, so no tokenizer model is involved.</summary>
public sealed class IndexTts2TextSegmenterTests
{
    private sealed record Case(string Text, string[] Tokens, int[] Ids, string[][] Segments, string[][] SegmentsQuick20, string[][] SegmentsMax24);

    private static Case[] LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "indextts2_segmenter_golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        List<Case> cases = [];
        foreach (JsonElement e in doc.RootElement.EnumerateArray())
        {
            string[] Strings(JsonElement a) => [.. a.EnumerateArray().Select(static x => x.GetString()!)];
            string[][] Nested(JsonElement a) => [.. a.EnumerateArray().Select(Strings)];
            cases.Add(new Case(
                e.GetProperty("text").GetString()!,
                Strings(e.GetProperty("tokens")),
                [.. e.GetProperty("ids").EnumerateArray().Select(static x => x.GetInt32())],
                Nested(e.GetProperty("segments")),
                Nested(e.GetProperty("segments_quick20")),
                Nested(e.GetProperty("segments_max24"))));
        }
        return [.. cases];
    }

    private static IndexTts2TextSegmenter.Token[] ToTokens(Case c) =>
        [.. c.Tokens.Select((t, i) => new IndexTts2TextSegmenter.Token(t, c.Ids[i]))];

    private static void AssertSegments(string[][] expected, List<IndexTts2TextSegmenter.Token[]> actual, string label)
    {
        Assert.True(expected.Length == actual.Count, $"{label}: expected {expected.Length} segments, got {actual.Count}.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i].SequenceEqual(actual[i].Select(static t => t.Piece)), $"{label}: segment {i} differs.");
    }

    [Fact]
    public void Split_MatchesTheReference_AtTheDefaultLimit()
    {
        foreach (Case c in LoadCases())
            AssertSegments(c.Segments, IndexTts2TextSegmenter.Split(ToTokens(c), 120), $"default [{c.Text[..Math.Min(30, c.Text.Length)]}]");
    }

    [Fact]
    public void Split_PreservesTheIdsOfEachPiece()
    {
        foreach (Case c in LoadCases())
        {
            IndexTts2TextSegmenter.Token[] tokens = ToTokens(c);
            foreach (IndexTts2TextSegmenter.Token[] seg in IndexTts2TextSegmenter.Split(tokens, 24))
                foreach (IndexTts2TextSegmenter.Token t in seg)
                    Assert.Contains(t, tokens);
        }
    }

    [Fact]
    public void Split_EmptyInput_YieldsNoSegments() =>
        Assert.Empty(IndexTts2TextSegmenter.Split([], 120));
}
