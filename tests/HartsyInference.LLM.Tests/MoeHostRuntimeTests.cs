using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Opt-in heterogeneous MoE path against the direct path on one tiny synthetic layer. CPU only.</summary>
public sealed class MoeHostRuntimeTests
{
    private const int Hidden = 8;
    private const int Intermediate = 6;
    private const int Experts = 4;
    private const int TopK = 2;
    private const int Tokens = 5;
    private const string Prefix = "blk.0";
    private readonly ITestOutputHelper _output;

    public MoeHostRuntimeTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptInPath_MatchesDirectPath_RoutesAndOutput(bool sharedExpert)
    {
        using CpuBackend backend = new();
        MoeConfig moe = Config(sharedExpert);
        Dictionary<string, Tensor> weights = SyntheticWeights(moe, seed: sharedExpert ? 7 : 3);
        Tensor x = Input(seed: 11);

        MoeFeedForward direct = new(moe, Hidden, lowVram: false);
        direct.LoadWeights(weights, Prefix);
        MoeFeedForward runtime = new(moe, Hidden, lowVram: false) { UseHostExpertRuntime = true };
        runtime.LoadWeights(weights, Prefix);

        Tensor expected = direct.Forward(backend, x, Tokens);
        Tensor actual = runtime.Forward(backend, x, Tokens);
        float maxAbs = MaxAbsDiff(expected, actual);
        _output.WriteLine($"shared={sharedExpert} max abs error {maxAbs:E3}");
        Assert.True(maxAbs <= 1e-5f, $"max abs error {maxAbs} exceeds 1e-5.");

        // Routes: the planner must serve exactly the experts the router picked, with the same number of (token, slot) pairs.
        Tensor logits = direct.ComputeRouterLogits(backend, x, Tokens);
        float[] logitValues = logits.AsReadOnlySpan<float>().ToArray();
        List<int>[] tokens = NewLists();
        List<float>[] routeWeights = NewFloatLists();
        direct.Route(logitValues, Tokens, Experts, TopK, tokens, routeWeights);
        ExpertAssignment[] plan = runtime.LastPlanForTest();
        int servedPairs = 0;
        int previous = -1;
        foreach (ExpertAssignment assignment in plan)
        {
            Assert.True(assignment.Key.Expert > previous, "Plan must list experts in ascending order.");
            previous = assignment.Key.Expert;
            Assert.Equal(ExpertPlacement.Cpu, assignment.Placement);
            Assert.Equal(tokens[assignment.Key.Expert].Count, assignment.Rows);
            servedPairs += assignment.Rows;
        }
        int routedExperts = tokens.Count(static list => list.Count > 0);
        Assert.Equal(routedExperts, plan.Length);
        Assert.Equal(Tokens * TopK, servedPairs);
    }

    [Fact]
    public void OptInPath_DefaultIsOff()
    {
        MoeFeedForward moe = new(Config(sharedExpert: false), Hidden, lowVram: false);
        Assert.False(moe.UseHostExpertRuntime);
    }

    [Fact]
    public void OptInPath_RejectsQuantizedExpertWeights()
    {
        using CpuBackend backend = new();
        MoeConfig moe = Config(sharedExpert: false);
        Dictionary<string, Tensor> weights = SyntheticWeights(moe, seed: 5);
        weights[$"{Prefix}.mlp.experts.0.gate_proj.weight"] = new Tensor(new TensorShape(Intermediate, Hidden), DType.F16);
        MoeFeedForward runtime = new(moe, Hidden, lowVram: false) { UseHostExpertRuntime = true };
        runtime.LoadWeights(weights, Prefix);
        Assert.Throws<NotSupportedException>(() => runtime.Forward(backend, Input(seed: 2), Tokens));
    }

    private static MoeConfig Config(bool sharedExpert) => new()
    {
        NumExperts = Experts,
        NumExpertsPerTok = TopK,
        MoeIntermediateSize = Intermediate,
        SharedExpertIntermediateSize = sharedExpert ? Intermediate : 0,
        Scoring = MoeScoring.Softmax,
        NormTopKProb = true,
    };

    private static Dictionary<string, Tensor> SyntheticWeights(MoeConfig moe, int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> weights = [];
        weights[$"{Prefix}.mlp.gate.weight"] = Random(random, moe.NumExperts, Hidden);
        for (int i = 0; i < moe.NumExperts; i++)
        {
            weights[$"{Prefix}.mlp.experts.{i}.gate_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.experts.{i}.up_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.experts.{i}.down_proj.weight"] = Random(random, Hidden, Intermediate);
        }
        if (moe.SharedExpertIntermediateSize > 0)
        {
            weights[$"{Prefix}.mlp.shared_expert.gate_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.shared_expert.up_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.shared_expert.down_proj.weight"] = Random(random, Hidden, Intermediate);
            weights[$"{Prefix}.mlp.shared_expert_gate.weight"] = Random(random, 1, Hidden);
        }
        return weights;
    }

    private static Tensor Random(Random random, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        Span<float> data = t.AsSpan<float>();
        for (int i = 0; i < data.Length; i++) data[i] = (float)(random.NextDouble() - 0.5) * 0.5f;
        return t;
    }

    private static Tensor Input(int seed) => Random(new Random(seed), Tokens, Hidden);

    private static float MaxAbsDiff(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        Assert.Equal(x.Length, y.Length);
        float max = 0f;
        for (int i = 0; i < x.Length; i++) max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        return max;
    }

    private static List<int>[] NewLists()
    {
        List<int>[] lists = new List<int>[Experts];
        for (int i = 0; i < Experts; i++) lists[i] = [];
        return lists;
    }

    private static List<float>[] NewFloatLists()
    {
        List<float>[] lists = new List<float>[Experts];
        for (int i = 0; i < Experts; i++) lists[i] = [];
        return lists;
    }
}
