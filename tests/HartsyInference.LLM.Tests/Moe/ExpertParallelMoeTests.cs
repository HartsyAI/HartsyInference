using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Tests.Kimi;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>The expert-parallel routed MoE over virtual ranks against the production single-block forward on the same weights. The token count
/// is not a multiple of the rank count, so every source rank holds a different number of tokens, and some rank sources none.</summary>
public sealed class ExpertParallelMoeTests
{
    private const float Tol = 1e-4f;
    private const string Prefix = "L0";

    [Theory]
    [InlineData(1, 7)]
    [InlineData(2, 7)]
    [InlineData(4, 7)]
    [InlineData(4, 2)]
    public void ExpertParallelForward_MatchesTheSingleBlock(int ranks, int tokens)
    {
        KimiK31Config c = KimiK31ShardedForwardTests.Config;
        Random rng = new(21);
        Dictionary<string, Tensor> whole = KimiK31ModelTests.MoeWeights(rng, c, Prefix);
        MoeConfig settings = KimiK31Model.MoeSettings(c) with { SharedExpertIntermediateSize = 0 };

        float[] input = new float[tokens * c.Hidden];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] expected = SingleBlock(settings, whole, input, tokens, c.Hidden);

        float[] actual = ExpertParallel(settings, whole, input, tokens, c.Hidden, ranks);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], Tol);
    }

    [Fact]
    public void ExpertParallel_RefusesAnExpertCountThatDoesNotSplit()
    {
        KimiK31Config c = KimiK31ShardedForwardTests.Config with { Experts = 6 };
        MoeConfig settings = KimiK31Model.MoeSettings(c) with { SharedExpertIntermediateSize = 0 };
        Dictionary<string, Tensor> whole = KimiK31ModelTests.MoeWeights(new Random(21), c, Prefix);
        List<IBackend> backends = [new CpuBackend(), new CpuBackend(), new CpuBackend(), new CpuBackend()];
        List<MoeFeedForward> blocks = [];
        for (int r = 0; r < 4; r++) blocks.Add(MakeBlock(settings, whole, c, r, 4, c.Hidden));
        ArgumentException ex = Assert.Throws<ArgumentException>(() => new ExpertParallelMoe(backends, blocks, new HostStagedComm(4), c.Experts, c.ExpertsPerToken));
        Assert.Contains("divide", ex.Message);
    }

    private static float[] SingleBlock(MoeConfig settings, Dictionary<string, Tensor> whole, float[] input, int tokens, int hidden)
    {
        MoeFeedForward block = new(settings, hidden, lowVram: false);
        block.LoadWeights(whole, Prefix);
        using Tensor x = new(new TensorShape(1, tokens, hidden), DType.F32);
        input.CopyTo(x.AsSpan<float>());
        using Tensor y = block.Forward(new CpuBackend(), x, tokens);
        return y.AsReadOnlySpan<float>().ToArray();
    }

    private static float[] ExpertParallel(MoeConfig settings, Dictionary<string, Tensor> whole, float[] input, int tokens, int hidden, int ranks)
    {
        KimiK31Config c = KimiK31ShardedForwardTests.Config;
        List<IBackend> backends = new(ranks);
        List<MoeFeedForward> blocks = new(ranks);
        for (int r = 0; r < ranks; r++)
        {
            backends.Add(new CpuBackend());
            blocks.Add(MakeBlock(settings, whole, c, r, ranks, hidden));
        }
        ICollectiveComm? comm = ranks > 1 ? new HostStagedComm(ranks) : null;
        ExpertParallelMoe moe = new(backends, blocks, comm, c.Experts, c.ExpertsPerToken);

        List<Tensor> inputs = new(ranks), accum = new(ranks);
        try
        {
            for (int r = 0; r < ranks; r++)
            {
                Tensor x = new(new TensorShape(1, tokens, hidden), DType.F32);
                input.CopyTo(x.AsSpan<float>());
                inputs.Add(x);
                accum.Add(new Tensor(new TensorShape(1, tokens, hidden), DType.F32));
            }
            moe.Forward(inputs, accum, tokens);
            comm?.AllReduceSum(accum);
            return accum[0].AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            foreach (Tensor t in inputs) t.Dispose();
            foreach (Tensor t in accum) t.Dispose();
            comm?.Dispose();
        }
    }

    // Rank r's block: the replicated router, latent projections and norms, and the experts of its contiguous range only.
    private static MoeFeedForward MakeBlock(MoeConfig settings, Dictionary<string, Tensor> whole, KimiK31Config c, int rank, int ranks, int hidden)
    {
        int per = c.Experts / ranks, first = rank * per;
        Dictionary<string, Tensor> block = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in whole)
        {
            if (!kv.Key.Contains(".experts.", StringComparison.Ordinal)) block[kv.Key] = kv.Value;
        }
        for (int g = first; g < first + per; g++)
        {
            foreach (string proj in new[] { "gate_proj", "up_proj", "down_proj" })
            {
                string key = $"{Prefix}.mlp.experts.{g}.{proj}.weight";
                block[key] = whole[key];
            }
        }
        MoeFeedForward moe = new(settings, hidden, lowVram: false);
        moe.ShardExperts(first, per);
        moe.LoadWeights(block, Prefix);
        return moe;
    }
}
