using System.Text.Json;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41IndexSelectionTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "candidate_select.json"))).RootElement;

    private static float[] Scores(JsonElement e) =>
        e.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? float.NegativeInfinity : v.GetSingle()).ToArray();

    [Fact]
    public void Candidate_Blocks_Match_Upstream()
    {
        int n = 0;
        foreach (JsonElement c in Fx.GetProperty("blocks").EnumerateArray())
        {
            float[] logits = Scores(c.GetProperty("logits"));
            bool[] expected = c.GetProperty("mask").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
            bool[] mask = new bool[logits.Length];
            DeepSeekV41IndexSelection.SelectCandidateBlocks(logits, c.GetProperty("compressLen").GetInt32(),
                c.GetProperty("topkBlocks").GetInt32(), c.GetProperty("blockSize").GetInt32(), mask);
            Assert.True(expected.SequenceEqual(mask), $"case {n++}");
        }
        Assert.True(n > 0);
    }

    [Fact]
    public void Index_TopK_Matches_Upstream()
    {
        int n = 0;
        foreach (JsonElement c in Fx.GetProperty("picks").EnumerateArray())
        {
            float[] scores = Scores(c.GetProperty("scores"));
            int[] expected = c.GetProperty("indices").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            int[] dest = new int[Math.Min(c.GetProperty("indexTopk").GetInt32(), scores.Length)];
            int written = DeepSeekV41IndexSelection.SelectTopK(scores, c.GetProperty("compressLen").GetInt32(),
                c.GetProperty("indexTopk").GetInt32(), c.GetProperty("offset").GetInt32(), dest);
            Assert.Equal(expected.Length, written);
            Assert.True(expected.SequenceEqual(dest), $"case {n++}: {string.Join(",", expected)} vs {string.Join(",", dest)}");
        }
        Assert.True(n > 0);
    }

    [Fact]
    public void Empty_Compressed_Range_Selects_Nothing_And_Pins_No_Block()
    {
        bool[] mask = new bool[8];
        DeepSeekV41IndexSelection.SelectCandidateBlocks(Enumerable.Repeat(float.NegativeInfinity, 8).ToArray(), 0, 2, 4, mask);
        Assert.All(mask, m => Assert.False(m));
    }
}
