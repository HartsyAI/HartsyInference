using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41HyperConnectionTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "hyper_connection.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-4f * Math.Max(1f, Math.Abs(expected[i])), $"{what}[{i}] {expected[i]} vs {actual[i]}");
    }

    [Theory]
    [InlineData("hc4")]
    [InlineData("hc2")]
    [InlineData("hc3_one_iter")]
    public void Mixes_Collapse_And_Expand_Match_Upstream_Block(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        int hc = c.GetProperty("hc").GetInt32(), dim = c.GetProperty("dim").GetInt32(), tokens = c.GetProperty("tokens").GetInt32();
        using CpuBackend cpu = new();
        DeepSeekV41HyperConnection conn = new(cpu, hc, dim, c.GetProperty("iters").GetInt32(), (float)c.GetProperty("hcEps").GetDouble(),
            (float)c.GetProperty("normEps").GetDouble(), Floats(c.GetProperty("fn")), Floats(c.GetProperty("scale")), Floats(c.GetProperty("base")));
        float[] x = Floats(c.GetProperty("x")), sub = Floats(c.GetProperty("sub"));
        float[] pre = new float[tokens * hc], post = new float[tokens * hc], comb = new float[tokens * hc * hc];

        conn.Mixes(x, tokens, pre, post, comb);
        AssertClose(Floats(c.GetProperty("pre")), pre, "pre");
        AssertClose(Floats(c.GetProperty("post")), post, "post");
        AssertClose(Floats(c.GetProperty("comb")), comb, "comb");

        float[] collapsed = new float[tokens * dim], expanded = new float[tokens * hc * dim];
        conn.Collapse(x, pre, tokens, collapsed);
        conn.Expand(sub, x, post, comb, tokens, expanded);
        AssertClose(Floats(c.GetProperty("collapsed")), collapsed, "collapsed");
        AssertClose(Floats(c.GetProperty("expanded")), expanded, "expanded");
    }

    [Fact]
    public void Rejects_Mismatched_Shapes()
    {
        using CpuBackend cpu = new();
        const int hc = 2, dim = 4;
        Assert.Throws<ArgumentException>(() => new DeepSeekV41HyperConnection(cpu, hc, dim, 3, 1e-6f, 1e-6f, new float[5], new float[3], new float[8]));
        DeepSeekV41HyperConnection conn = new(cpu, hc, dim, 3, 1e-6f, 1e-6f, new float[8 * hc * dim], new float[3], new float[8]);
        Assert.Throws<ArgumentException>(() => conn.Mixes(new float[hc * dim - 1], 1, new float[hc], new float[hc], new float[hc * hc]));
        Assert.Throws<ArgumentException>(() => conn.Collapse(new float[hc * dim], new float[hc], 1, new float[dim + 1]));
    }

    [Fact]
    public void Rejects_Out_Of_Range_Iterations_And_Epsilons()
    {
        using CpuBackend cpu = new();
        const int hc = 2, dim = 4;
        float[] fn = new float[8 * hc * dim], scale = new float[3], bias = new float[8];
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41HyperConnection(cpu, hc, dim, 0, 1e-6f, 1e-6f, fn, scale, bias));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41HyperConnection(cpu, hc, dim, 3, -1f, 1e-6f, fn, scale, bias));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41HyperConnection(cpu, hc, dim, 3, 1e-6f, 0f, fn, scale, bias));
    }
}
