using System.Text.Json;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41CompressorStateTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "compressor_pool.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static float[] Project(float[] x, float[] w, int rows, int dim, int outDim)
    {
        float[] y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outDim; o++)
            {
                float s = 0f;
                for (int k = 0; k < dim; k++) s += x[r * dim + k] * w[o * dim + k];
                y[r * outDim + o] = s;
            }
        return y;
    }

    [Theory]
    [InlineData("r4_remainder")]
    [InlineData("r2_odd")]
    public void Prefill_Then_Decode_Matches_Upstream_Compressor(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        int ratio = c.GetProperty("ratio").GetInt32(), dim = c.GetProperty("dim").GetInt32(), head = c.GetProperty("headDim").GetInt32();
        float[] x = Floats(c.GetProperty("x")), wkv = Floats(c.GetProperty("wkv")), wgate = Floats(c.GetProperty("wgate"));
        DeepSeekV41CompressorState state = new(ratio, head);
        int consumed = 0;
        foreach (JsonElement step in c.GetProperty("steps").EnumerateArray())
        {
            int len = step.GetProperty("len").GetInt32(), start = step.GetProperty("startPos").GetInt32();
            float[] xs = x.AsSpan(consumed * dim, len * dim).ToArray();
            consumed += len;
            float[] dest = new float[state.MaxRows(start, len) * head];
            int rows = state.Pool(Project(xs, wkv, len, dim, head), Project(xs, wgate, len, dim, head), len, start, dest);
            Assert.Equal(step.GetProperty("rows").GetInt32(), rows);
            if (rows == 0) continue;
            float[] expected = Floats(step.GetProperty("out"));
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - dest[i]) < 1e-4f, $"step {start} [{i}] {expected[i]} vs {dest[i]}");
        }
    }

    [Fact]
    public void Chunked_Prefill_Equals_One_Shot_Prefill()
    {
        const int ratio = 4, head = 6, total = 13;
        Random rng = new(3);
        float[] kv = Enumerable.Range(0, total * head).Select(_ => (float)rng.NextDouble()).ToArray();
        float[] score = Enumerable.Range(0, total * head).Select(_ => (float)rng.NextDouble()).ToArray();
        DeepSeekV41CompressorState whole = new(ratio, head), chunked = new(ratio, head);
        float[] a = new float[whole.MaxRows(0, total) * head];
        int rowsA = whole.Pool(kv, score, total, 0, a);
        List<float> b = [];
        foreach ((int s, int n) in new[] { (0, 5), (5, 3), (8, 5) })
        {
            float[] part = new float[chunked.MaxRows(s, n) * head];
            int rows = chunked.Pool(kv.AsSpan(s * head, n * head), score.AsSpan(s * head, n * head), n, s, part);
            b.AddRange(part.Take(rows * head));
        }
        Assert.Equal(rowsA * head, b.Count);
        Assert.Equal(a.Take(rowsA * head), b);
    }
}
