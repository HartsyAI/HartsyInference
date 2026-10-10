using System.Text.Json;
using HartsyInference.LLM.DeepSeekV41.Engram;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41.Engram;

public sealed class DeepSeekV41EngramModuleTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "engram_module.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static ushort Bf16(float v) => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);

    private static (DeepSeekV41EngramModule Module, float[] X, long[] Ids, bool[] Mask, float[] Expected, int Tokens) Build(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        int dim = c.GetProperty("dim").GetInt32(), hc = c.GetProperty("hc").GetInt32(), cols = c.GetProperty("cols").GetInt32();
        int headDim = c.GetProperty("headDim").GetInt32();
        ushort[] table = Floats(c.GetProperty("table")).Select(Bf16).ToArray();
        void Gather(ReadOnlySpan<long> rows, Span<ushort> dest)
        {
            for (int r = 0; r < rows.Length; r++) table.AsSpan((int)rows[r] * headDim, headDim).CopyTo(dest.Slice(r * headDim, headDim));
        }
        DeepSeekV41EngramModule module = new(dim, hc, cols, headDim, (float)c.GetProperty("eps").GetDouble(), Floats(c.GetProperty("wkv")),
            Floats(c.GetProperty("qWeight")), Floats(c.GetProperty("kWeight")), Gather);
        bool[] mask = c.GetProperty("mask").ValueKind == JsonValueKind.Null ? [] : c.GetProperty("mask").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
        return (module, Floats(c.GetProperty("x")), c.GetProperty("hashIds").EnumerateArray().Select(v => v.GetInt64()).ToArray(), mask,
            Floats(c.GetProperty("y")), c.GetProperty("tokens").GetInt32());
    }

    [Theory]
    [InlineData("plain")]
    [InlineData("masked")]
    public void Output_Matches_Upstream_Engram_Forward(string name)
    {
        (DeepSeekV41EngramModule module, float[] x, long[] ids, bool[] mask, float[] expected, int tokens) = Build(name);
        module.Apply(x, tokens, ids, mask);
        for (int i = 0; i < x.Length; i++)
            Assert.True(Math.Abs(expected[i] - x[i]) <= 2e-4f * Math.Max(1f, Math.Abs(expected[i])), $"[{i}] {expected[i]} vs {x[i]}");
    }

    [Fact]
    public void Masked_Positions_Are_Left_Untouched()
    {
        (DeepSeekV41EngramModule module, float[] x, long[] ids, bool[] mask, _, int tokens) = Build("masked");
        float[] before = (float[])x.Clone();
        module.Apply(x, tokens, ids, mask);
        int stride = x.Length / tokens;
        for (int t = 0; t < tokens; t++)
            if (!mask[t]) Assert.Equal(before.AsSpan(t * stride, stride).ToArray(), x.AsSpan(t * stride, stride).ToArray());
    }
}
