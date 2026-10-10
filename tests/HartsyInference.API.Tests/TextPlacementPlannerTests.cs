using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Placement;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The text placement planner on synthetic devices: the order Auto tries, what a forced mode refuses, and the header
/// reading its demand comes from. A wrong answer here is an out-of-memory error partway through a load, or a slow placement chosen
/// when a fast one fit, and neither fails a test that does not check the decision itself.</summary>
public sealed class TextPlacementPlannerTests
{
    private const long Gb = 1L << 30;

    /// <summary>Qwen3-30B-A3B Q4_K_M in round numbers: 1 GB dense, 17.5 GB experts, 48 layers, 192 KiB of F32 KV per token.</summary>
    private static readonly TextPlacementDemand Qwen30B = new(1 * Gb, 35 * Gb / 2, 48, 196_608, 8192);

    private static readonly TextPlacementDemand Dense8B = new(8 * Gb, 0, 32, 131_072, 8192);

    private static TextPlacementDevice Card(string device, double freeGb) => new(device, (long)(freeGb * Gb));

    [Fact]
    public void Auto_PrefersOneGpu_WhenItFits()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:0", 23.4), Card("cuda:1", 11.6)], 50 * Gb,
            TextPlacementMode.Auto);
        Assert.True(plan.Feasible);
        Assert.Equal(TextPlacementMode.Gpu, plan.Mode);
        Assert.Equal(["cuda:0"], plan.Devices);
    }

    [Fact]
    public void Auto_SplitsAcrossGpus_WhenOneIsTooSmall()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:1", 11.6), Card("cuda:0", 23.4)], 50 * Gb,
            TextPlacementMode.Auto);
        Assert.True(plan.Feasible);
        Assert.Equal(TextPlacementMode.Split, plan.Mode);
        Assert.Equal("cuda:1+cuda:0", plan.DeviceKey);
    }

    [Fact]
    public void Auto_OffloadsExperts_WhenNoSplitFits()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:0", 11.6)], 50 * Gb, TextPlacementMode.Auto);
        Assert.True(plan.Feasible);
        Assert.Equal(TextPlacementMode.Offload, plan.Mode);
        Assert.InRange(plan.ExpertBudgetBytes, TextPlacementPlanner.MinExpertBudgetBytes, Qwen30B.ExpertBytes - 1);
        // Dense weights, KV, reserve and the expert cache must fit what the device has.
        Assert.True(plan.RequiredBytes <= Card("cuda:0", 11.6).FreeBytes);
    }

    [Fact]
    public void Auto_IsInfeasible_WhenNotEvenTheDenseWeightsFit_AndNamesEveryAccount()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:0", 2.5)], 50 * Gb, TextPlacementMode.Auto);
        Assert.False(plan.Feasible);
        Assert.Contains("One GPU:", plan.Reason);
        Assert.Contains("Split:", plan.Reason);
        Assert.Contains("Offload:", plan.Reason);
    }

    [Fact]
    public void Offload_IsRefused_WhenHostRamCannotHoldTheExpertsLeftOnTheCpu()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:0", 6)], 4 * Gb, TextPlacementMode.Offload);
        Assert.False(plan.Feasible);
        Assert.Contains("host RAM", plan.Reason);
    }

    [Fact]
    public void Offload_IsNeverChosenForADenseModel()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Dense8B, [Card("cuda:0", 6)], 50 * Gb, TextPlacementMode.Auto);
        Assert.False(plan.Feasible);
        TextPlacement forced = TextPlacementPlanner.Plan(Dense8B, [Card("cuda:0", 24)], 50 * Gb, TextPlacementMode.Offload);
        Assert.False(forced.Feasible);
        Assert.Contains("dense", forced.Reason);
    }

    [Fact]
    public void ForcedGpu_RefusesInsteadOfFallingBack()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Qwen30B, [Card("cuda:0", 11.6), Card("cuda:1", 23.4)], 50 * Gb,
            TextPlacementMode.Gpu);
        Assert.False(plan.Feasible);
        Assert.Equal(TextPlacementMode.Gpu, plan.Mode);
        Assert.Equal(["cuda:0"], plan.Devices);
    }

    [Fact]
    public void ForcedSplit_UsesEveryDevice_EvenWhenOneWouldDo()
    {
        TextPlacement plan = TextPlacementPlanner.Plan(Dense8B, [Card("cuda:0", 23.4), Card("cuda:1", 11.6)], 50 * Gb,
            TextPlacementMode.Split);
        Assert.True(plan.Feasible);
        Assert.Equal(["cuda:0", "cuda:1"], plan.Devices);
    }

    [Fact]
    public void Split_ChargesAReservePerDevice()
    {
        // 18.5 GB + 1.5 GB KV fits 21.6 GB with one reserve but not with two.
        TextPlacementDemand demand = new(1 * Gb, (long)(17.5 * Gb), 48, 196_608, 8192);
        long oneReserve = demand.TotalBytes + TextPlacementPlanner.ReserveBytes;
        TextPlacement plan = TextPlacementPlanner.Plan(demand, [new("cuda:0", oneReserve / 2), new("cuda:1", oneReserve / 2 + 1)],
            null, TextPlacementMode.Split);
        Assert.False(plan.Feasible);
    }

    [Theory]
    [InlineData(null, TextPlacementMode.Auto)]
    [InlineData("", TextPlacementMode.Auto)]
    [InlineData("AUTO", TextPlacementMode.Auto)]
    [InlineData(" gpu ", TextPlacementMode.Gpu)]
    [InlineData("Split", TextPlacementMode.Split)]
    [InlineData("offload", TextPlacementMode.Offload)]
    public void Modes_ParseTheirSpellings(string? value, TextPlacementMode expected) =>
        Assert.Equal(expected, TextPlacementModes.Parse(value));

    [Fact]
    public void Modes_RefuseAMisspelling() =>
        Assert.Throws<HartsyInferenceException>(() => TextPlacementModes.Parse("offlaod"));

    [Fact]
    public void Header_SeparatesExperts_LeavesAnUntiedEmbeddingOnTheHost_AndWidensUnreadableQuants()
    {
        TextTensorInfo[] tensors =
        [
            new("token_embd.weight", DType.Q4_K, 1024 * 256),
            new("output.weight", DType.Q6_K, 1024 * 256),
            new("blk.0.attn_q.weight", DType.Q4_K, 256 * 256),
            new("blk.0.attn_norm.weight", DType.F32, 256),
            new("blk.0.ffn_gate_exps.weight", DType.Q4_K, 8 * 256 * 512),
            new("blk.0.ffn_down_exps.weight", DType.Q6_K, 8 * 512 * 256),
            new("blk.0.attn_k.weight", DType.IQ4_XS, 256 * 256),
        ];
        Dictionary<string, long> arch = new()
        {
            ["block_count"] = 1, ["attention.head_count"] = 4, ["attention.head_count_kv"] = 2, ["embedding_length"] = 256,
        };
        TextPlacementDemand demand = TextPlacementDemandReader.FromHeader(tensors, k => arch.TryGetValue(k, out long v) ? v : null,
            contextTokens: 100, includeRedundantSplits: false, kvF16: false);

        long expectedDense = DType.Q6_K.ComputeByteCount(1024 * 256) + DType.Q4_K.ComputeByteCount(256 * 256) + 256 * 4
            + 256 * 256 * 4; // IQ4_XS is not device-readable, so it is widened to F32
        Assert.Equal(expectedDense, demand.DenseBytes);
        Assert.Equal(DType.Q4_K.ComputeByteCount(8 * 256 * 512) + DType.Q6_K.ComputeByteCount(8 * 512 * 256), demand.ExpertBytes);
        // 1 layer x 2 KV heads x (64 + 64) x 4 bytes.
        Assert.Equal(1L * 2 * 128 * 4, demand.KvBytesPerToken);
        Assert.Equal(100 * demand.KvBytesPerToken, demand.KvBytes);

        TextPlacementDemand withSplits = TextPlacementDemandReader.FromHeader(tensors, k => arch.TryGetValue(k, out long v) ? v : null,
            contextTokens: 100, includeRedundantSplits: true, kvF16: true);
        Assert.Equal(expectedDense + DType.Q4_K.ComputeByteCount(256 * 256) + 256 * 256 * 4, withSplits.DenseBytes);
        Assert.Equal(demand.KvBytesPerToken / 2, withSplits.KvBytesPerToken);
    }

    [Fact]
    public void Header_TiedEmbedding_IsCountedOnTheDevice()
    {
        TextTensorInfo[] tensors = [new("token_embd.weight", DType.Q8_0, 1024 * 256)];
        TextPlacementDemand demand = TextPlacementDemandReader.FromHeader(tensors, _ => null, 1, false, false);
        Assert.Equal(DType.Q8_0.ComputeByteCount(1024 * 256), demand.DenseBytes);
    }
}
