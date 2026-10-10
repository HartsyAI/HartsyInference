using System.Text.Json;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41RopeTableTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "rope_table.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    [Theory]
    [InlineData("plain")]
    [InlineData("yarn")]
    public void Table_Matches_Upstream_Precompute_Freqs_Cis(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        int original = c.GetProperty("originalSeqLen").GetInt32();
        DeepSeekV41RopeScaling? scaling = original > 0
            ? new DeepSeekV41RopeScaling(c.GetProperty("factor").GetDouble(), c.GetProperty("betaFast").GetDouble(),
                c.GetProperty("betaSlow").GetDouble(), original)
            : null;
        DeepSeekV41RopeTable t = DeepSeekV41RopeTable.Build(
            c.GetProperty("rotaryDim").GetInt32(), c.GetProperty("length").GetInt32(), c.GetProperty("theta").GetDouble(), scaling);
        float[] cos = Floats(c.GetProperty("cos")), sin = Floats(c.GetProperty("sin"));
        Assert.Equal(cos.Length, t.Cos.Length);
        for (int i = 0; i < cos.Length; i++)
        {
            Assert.True(Math.Abs(cos[i] - t.Cos[i]) < 2e-5f, $"cos[{i}] {cos[i]} vs {t.Cos[i]}");
            Assert.True(Math.Abs(sin[i] - t.Sin[i]) < 2e-5f, $"sin[{i}] {sin[i]} vs {t.Sin[i]}");
        }
    }
}
