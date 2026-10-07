using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41MoeExecutorTests
{
    private static readonly JsonElement Fx = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "moe_exec.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private sealed class ListSource(DeepSeekV41SwigluWeights[] experts) : IDeepSeekV41ExpertSource
    {
        public List<int> Requested { get; } = [];

        public DeepSeekV41SwigluWeights GetExpert(int expert)
        {
            Requested.Add(expert);
            return experts[expert];
        }
    }

    private static (ListSource Source, DeepSeekV41SwigluWeights Shared) Weights(JsonElement c)
    {
        int dim = c.GetProperty("dim").GetInt32(), inter = c.GetProperty("inter").GetInt32();
        DeepSeekV41SwigluWeights[] experts = Enumerable.Range(0, c.GetProperty("experts").GetInt32()).Select(i =>
            new DeepSeekV41SwigluWeights(dim, inter, Floats(c.GetProperty("w1")[i]), Floats(c.GetProperty("w2")[i]),
                Floats(c.GetProperty("w3")[i]))).ToArray();
        DeepSeekV41SwigluWeights shared = new(dim, inter, Floats(c.GetProperty("sw1")), Floats(c.GetProperty("sw2")),
            Floats(c.GetProperty("sw3")));
        return (new ListSource(experts), shared);
    }

    [Theory]
    [InlineData("limit")]
    [InlineData("nolimit")]
    [InlineData("top1")]
    public void Output_Matches_Upstream_MoE_Forward(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        (ListSource source, DeepSeekV41SwigluWeights shared) = Weights(c);
        int tokens = c.GetProperty("tokens").GetInt32(), dim = c.GetProperty("dim").GetInt32();
        float[] y = new float[tokens * dim];
        DeepSeekV41MoeExecutor.Run(Floats(c.GetProperty("x")), tokens,
            c.GetProperty("indices").EnumerateArray().Select(v => v.GetInt32()).ToArray(), Floats(c.GetProperty("weights")),
            c.GetProperty("topk").GetInt32(), c.GetProperty("experts").GetInt32(), source, shared,
            (float)c.GetProperty("limit").GetDouble(), y);
        float[] expected = Floats(c.GetProperty("y"));
        for (int i = 0; i < y.Length; i++)
            Assert.True(Math.Abs(expected[i] - y[i]) <= 2e-4f * Math.Max(1f, Math.Abs(expected[i])), $"[{i}] {expected[i]} vs {y[i]}");
    }

    [Fact]
    public void Only_Routed_Experts_Are_Requested_Once_Each()
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == "limit");
        (ListSource source, DeepSeekV41SwigluWeights shared) = Weights(c);
        int tokens = c.GetProperty("tokens").GetInt32(), dim = c.GetProperty("dim").GetInt32();
        int[] idx = c.GetProperty("indices").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        DeepSeekV41MoeExecutor.Run(Floats(c.GetProperty("x")), tokens, idx, Floats(c.GetProperty("weights")),
            c.GetProperty("topk").GetInt32(), c.GetProperty("experts").GetInt32(), source, shared,
            (float)c.GetProperty("limit").GetDouble(), new float[tokens * dim]);
        Assert.Equal(idx.Distinct().OrderBy(i => i), source.Requested);
    }

    [Fact]
    public void Out_Of_Range_Expert_Ids_Are_Skipped()
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == "top1");
        (ListSource source, DeepSeekV41SwigluWeights shared) = Weights(c);
        int tokens = 1, dim = c.GetProperty("dim").GetInt32();
        float[] x = Floats(c.GetProperty("x")).Take(dim).ToArray();
        float[] withRoute = new float[dim], onlyShared = new float[dim];
        DeepSeekV41MoeExecutor.Run(x, tokens, [-1], [1f], 1, 3, source, shared, 2f, withRoute);
        DeepSeekV41MoeExecutor.Run(x, tokens, [99], [1f], 1, 3, source, shared, 2f, onlyShared);
        Assert.Equal(onlyShared, withRoute);
        Assert.Empty(source.Requested);
    }

    [Theory]
    [InlineData("limit")]
    [InlineData("nolimit")]
    [InlineData("top1")]
    public void Whole_Layer_Including_The_Gate_Matches_Upstream_MoE(string name)
    {
        JsonElement c = Fx.GetProperty("cases").EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        (ListSource source, DeepSeekV41SwigluWeights shared) = Weights(c);
        int tokens = c.GetProperty("tokens").GetInt32(), dim = c.GetProperty("dim").GetInt32(), experts = c.GetProperty("experts").GetInt32();
        int k = c.GetProperty("topk").GetInt32();
        MoeRouteArgs route = new(experts, k, MoeRouteScoring.SqrtSoftplus, Renormalize: k > 1, RenormEpsilon: 1e-20f);
        using CpuBackend cpu = new();
        DeepSeekV41MoeLayer layer = new(cpu, Floats(c.GetProperty("gateWeight")), Floats(c.GetProperty("gateBias")), null, route, source,
            shared, (float)c.GetProperty("limit").GetDouble());
        float[] y = new float[tokens * dim];
        layer.Forward(Floats(c.GetProperty("x")), tokens, default, y);
        float[] expected = Floats(c.GetProperty("y"));
        for (int i = 0; i < y.Length; i++)
            Assert.True(Math.Abs(expected[i] - y[i]) <= 2e-4f * Math.Max(1f, Math.Abs(expected[i])), $"[{i}] {expected[i]} vs {y[i]}");
    }
}
