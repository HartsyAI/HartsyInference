using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Moe.Residency;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>
/// Expert offload on the CPU lane: a <see cref="HostExpertCache"/> stands in for the device cache (its residents run through the
/// backend's projections, exactly as cached experts do on CUDA) and a reference runner stands in for the packed CPU kernels. The
/// weights are F32, so a correct split reproduces the direct path to rounding. A wrong one — a row combined twice, an expert run on
/// the wrong rows, a miss dropped — is a silent numerical error, which is what these hold against.
/// </summary>
public sealed class MoeExpertOffloadTests(ITestOutputHelper output)
{
    private const int Hidden = 32;
    private const int Intermediate = 32;
    private const int Experts = 8;
    private const int TopK = 2;
    private const string Prefix = "blk.0";

    private static MoeConfig Config() => new()
    {
        NumExperts = Experts,
        NumExpertsPerTok = TopK,
        MoeIntermediateSize = Intermediate,
        Scoring = MoeScoring.Softmax,
        NormTopKProb = true,
    };

    /// <summary>Runs CPU experts with the F32 reference on the offload's registered weights.</summary>
    private sealed class ReferenceRunner(Func<MoeExpertOffload> offload) : IExpertHostRunner
    {
        public int Calls { get; private set; }

        public void Run(ExpertProgram program, ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y)
        {
            Calls++;
            ExpertWeights w = offload().Resolve(key);
            F32ExpertWeights f32 = new(Hidden, Intermediate, w.W1.Weight.AsReadOnlySpan<float>().ToArray(),
                w.W3.Weight.AsReadOnlySpan<float>().ToArray(), w.W2.Weight.AsReadOnlySpan<float>().ToArray());
            ExpertProgramReference.Apply(program, f32, x, rows, y);
        }
    }

    private static (MoeExpertOffload Offload, ReferenceRunner Runner) NewOffload(int residentExperts, int streamRowThreshold = 1 << 20)
    {
        long expertBytes = 3L * Intermediate * Hidden * sizeof(float);
        MoeExpertOffload? offload = null;
        ReferenceRunner runner = new(() => offload!);
        offload = new MoeExpertOffload(new HostExpertCache(Math.Max(1, residentExperts * expertBytes)), runner,
            new DecayedLfuHysteresisPolicy(0.999, 0.25)) { StreamRowThreshold = streamRowThreshold };
        return (offload, runner);
    }

    [Theory]
    [InlineData(0, 1)]   // every expert on the CPU, decode
    [InlineData(3, 1)]   // a mixed split, decode
    [InlineData(3, 7)]   // a mixed split, a prefill-sized batch
    [InlineData(8, 7)]   // every expert resident
    public void OffloadMatchesTheDirectPath(int residentExperts, int tokens)
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = SyntheticWeights(seed: 11);
        Tensor x = Random(new Random(5), tokens, Hidden);

        MoeFeedForward direct = new(Config(), Hidden, lowVram: false);
        direct.LoadWeights(weights, Prefix);
        Tensor expected = direct.Forward(backend, x, tokens);

        (MoeExpertOffload offload, ReferenceRunner runner) = NewOffload(residentExperts);
        using (offload)
        {
            MoeFeedForward routed = new(Config(), Hidden, lowVram: false);
            routed.LoadWeights(weights, Prefix);
            routed.AttachOffload(offload, layer: 0);
            offload.Seed([(0, Experts)], residentExperts);
            Tensor actual = routed.Forward(backend, x, tokens);

            float maxAbs = MaxAbsDiff(expected, actual);
            MoeOffloadStats stats = offload.Stats;
            output.WriteLine($"resident={residentExperts} tokens={tokens} maxAbs={maxAbs:E3} {stats} cpuCalls={runner.Calls}");
            Assert.True(maxAbs <= 1e-5f, $"max abs error {maxAbs}");
            // Every routed (token, slot) pair ran exactly once, somewhere.
            Assert.Equal(tokens * TopK, stats.ResidentRows + stats.StreamedRows + stats.HostRows);
            if (residentExperts == 0) Assert.Equal(0, stats.ResidentRows);
            if (residentExperts == Experts) Assert.Equal(0, stats.HostRows);
        }
    }

    [Fact]
    public void LargeBatchMissesStream_InsteadOfRunningOnTheCpu()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = SyntheticWeights(seed: 3);
        const int tokens = 16;
        Tensor x = Random(new Random(9), tokens, Hidden);
        MoeFeedForward direct = new(Config(), Hidden, lowVram: false);
        direct.LoadWeights(weights, Prefix);
        Tensor expected = direct.Forward(backend, x, tokens);

        (MoeExpertOffload offload, ReferenceRunner runner) = NewOffload(residentExperts: 0, streamRowThreshold: 1);
        using (offload)
        {
            MoeFeedForward routed = new(Config(), Hidden, lowVram: false);
            routed.LoadWeights(weights, Prefix);
            routed.AttachOffload(offload, layer: 0);
            Tensor actual = routed.Forward(backend, x, tokens);
            Assert.True(MaxAbsDiff(expected, actual) <= 1e-5f);
            Assert.Equal(0, runner.Calls);
            Assert.Equal(tokens * TopK, offload.Stats.StreamedRows);
        }
    }

    [Fact]
    public void Admission_FillsFreeRoomWithoutEvicting_ThenSwapsOnlyHotExperts()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = SyntheticWeights(seed: 21);
        (MoeExpertOffload offload, _) = NewOffload(residentExperts: 2);
        using (offload)
        {
            MoeFeedForward routed = new(Config(), Hidden, lowVram: false);
            routed.LoadWeights(weights, Prefix);
            routed.AttachOffload(offload, layer: 0);

            // An empty cache with room for two experts takes at most two misses and evicts nothing.
            routed.Forward(backend, Random(new Random(1), 6, Hidden), 6);
            ExpertCacheStats first = offload.CacheBase.Stats;
            Assert.Equal(2, first.ResidentExperts);
            Assert.Equal(0, first.Evictions);

            // Full: a cold miss (scored below the threshold) is never admitted, so nothing more is uploaded.
            MoeExpertOffload strict = offload;
            long before = strict.Stats.Admitted;
            routed.Forward(backend, Random(new Random(2), 1, Hidden), 1);
            Assert.True(strict.Stats.Admitted - before <= strict.MaxSwapsPerLayer);
            Assert.True(offload.CacheBase.Stats.ResidentBytes <= offload.CacheBase.Stats.BudgetBytes);
        }
    }

    [Fact]
    public void Reload_DetachesTheOffload()
    {
        (MoeExpertOffload offload, _) = NewOffload(residentExperts: 1);
        using (offload)
        {
            MoeFeedForward routed = new(Config(), Hidden, lowVram: false);
            routed.LoadWeights(SyntheticWeights(seed: 1), Prefix);
            routed.AttachOffload(offload, layer: 0);
            Assert.Same(offload, routed.Offload);
            routed.LoadWeights(SyntheticWeights(seed: 2), Prefix);
            Assert.Null(routed.Offload);
        }
    }

    private static Dictionary<string, Tensor> SyntheticWeights(int seed)
    {
        Random random = new(seed);
        Dictionary<string, Tensor> weights = [];
        weights[$"{Prefix}.mlp.gate.weight"] = Random(random, Experts, Hidden);
        for (int i = 0; i < Experts; i++)
        {
            weights[$"{Prefix}.mlp.experts.{i}.gate_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.experts.{i}.up_proj.weight"] = Random(random, Intermediate, Hidden);
            weights[$"{Prefix}.mlp.experts.{i}.down_proj.weight"] = Random(random, Hidden, Intermediate);
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

    private static float MaxAbsDiff(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        Assert.Equal(x.Length, y.Length);
        float max = 0f;
        for (int i = 0; i < x.Length; i++) max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        return max;
    }
}
