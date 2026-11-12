using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The tensor- and expert-parallel decoder on virtual CPU ranks (<see cref="HostStagedComm"/>) must match the single-rank decoder on the
/// same weights. Heads, vocabulary, experts and the dense and shared intermediates all split evenly at 2 and 4 ranks.</summary>
public sealed class KimiK31ShardedForwardTests
{
    private const float Tol = 1e-4f;

    // 4 heads, 8 experts and the other split dimensions divide by both 2 and 4 ranks.
    internal static readonly KimiK31Config Config = KimiK31ModelTests.Config with
    {
        Heads = 4, Experts = 8, ExpertsPerToken = 2, FullAttentionLayers = new HashSet<int> { 1, 3 },
    };

    private static readonly int[] Tokens = [4, 4, 9, 0, 13, 7];

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void ShardedForward_MatchesSingleRank(int ranks)
    {
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9), Config);
        float[] single = RunSingle(weights);

        float[] sharded = RunSharded(weights, ranks);

        Assert.Equal(single.Length, sharded.Length);
        for (int i = 0; i < single.Length; i++) Assert.Equal(single[i], sharded[i], Tol);
    }

    [Fact]
    public void ShardedState_CarriesEachRanksCachesAcrossTokens()
    {
        // A second call advances the same sharded state: its logits depend on the earlier tokens through every rank's caches.
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9), Config);
        using KimiK31Model model = Sharded(weights, 2);
        using KimiK31State state = new(Config, capacity: 16, ranks: 2);
        model.Forward([4], state);
        float[] withHistory = model.Forward([1], state);
        using KimiK31Model fresh = Sharded(weights, 2);
        using KimiK31State freshState = new(Config, capacity: 16, ranks: 2);
        float[] alone = fresh.Forward([1], freshState);

        Assert.Equal(2, state.Position);
        bool differs = false;
        for (int i = 0; i < alone.Length; i++) differs |= MathF.Abs(withHistory[i] - alone[i]) > 1e-5f;
        Assert.True(differs, "the sharded state did not carry the earlier token");
    }

    [Fact]
    public void Model_RefusesAHeadCountThatDoesNotSplit()
    {
        // The synthetic decoder has 2 heads; 4 ranks cannot each own a whole head.
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9));
        ArgumentException ex = Assert.Throws<ArgumentException>(() => Sharded(weights, KimiK31ModelTests.Config, 4));
        Assert.Contains("Heads", ex.Message);
    }

    [Fact]
    public void Require_RefusesAnExpertCountThatDoesNotSplit()
    {
        KimiK31Config nine = Config with { Experts = 9 };
        ArgumentException ex = Assert.Throws<ArgumentException>(() => KimiK31Sharding.Require(nine, 4));
        Assert.Contains("Experts", ex.Message);
    }

    [Fact]
    public void State_RefusesAModelOfAnotherRankCount()
    {
        KimiK31Weights weights = KimiK31ModelTests.RandomWeights(new Random(9), Config);
        using KimiK31Model model = Sharded(weights, 2);
        using KimiK31State state = new(Config, capacity: 16);
        Assert.Throws<ArgumentException>(() => model.ForwardToken(1, state));
    }

    private static float[] RunSingle(KimiK31Weights weights)
    {
        using KimiK31Model model = new(new CpuBackend(), Config, weights, maxPositions: 16);
        using KimiK31State state = new(Config, capacity: 16);
        return model.Forward(Tokens, state);
    }

    private static float[] RunSharded(KimiK31Weights weights, int ranks)
    {
        using KimiK31Model model = Sharded(weights, ranks);
        using KimiK31State state = new(Config, capacity: 16, ranks);
        return model.Forward(Tokens, state);
    }

    private static KimiK31Model Sharded(KimiK31Weights weights, int ranks) => Sharded(weights, Config, ranks);

    private static KimiK31Model Sharded(KimiK31Weights weights, KimiK31Config config, int ranks)
    {
        List<IBackend> backends = new(ranks);
        for (int r = 0; r < ranks; r++) backends.Add(new CpuBackend());
        TpPlacement placement = new(backends, new HostStagedComm(ranks));
        return new KimiK31Model(placement, config, weights, maxPositions: 16);
    }
}
