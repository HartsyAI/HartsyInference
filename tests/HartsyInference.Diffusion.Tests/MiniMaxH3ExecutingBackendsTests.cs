using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Which devices H3 asks before deciding whether a quantized DiT can stay packed. Asking one that never runs
/// a block widens the whole checkpoint on the host for nothing — a 6.7 GB Q2_K build becomes roughly 40 GB — so the
/// set has to be the devices that execute, not every device that was configured.</summary>
public sealed class MiniMaxH3ExecutingBackendsTests
{
    private static RecipeContext Context(IBackend primary, IBackend? cfg = null, IBackend? shard = null,
        IReadOnlyList<IBackend>? cp = null) =>
        new RecipeContext
        {
            CheckpointPath = "dit.gguf",
            Backend = primary,
            CfgParallelBackend = cfg,
            DitShardBackend = shard,
            CpBackends = cp,
        };

    [Fact]
    public void PrimaryAlone_WhenNothingElseIsConfigured()
    {
        using CpuBackend primary = new CpuBackend();
        Assert.Equal([primary], MiniMaxH3Recipe.ExecutingBackends(Context(primary)).ToArray());
    }

    /// <summary>The shard peer does run blocks, so it counts — even though whether sharding ends up enabled is not
    /// settled until after the weights have been prepared.</summary>
    [Fact]
    public void IncludesTheDitShardPeer()
    {
        using CpuBackend primary = new CpuBackend();
        using CpuBackend shard = new CpuBackend();
        Assert.Equal([primary, shard], MiniMaxH3Recipe.ExecutingBackends(Context(primary, shard: shard)).ToArray());
    }

    /// <summary>The regression this exists for. <c>WarnIfPlacementIgnored</c> tells the operator that H3 runs on
    /// neither of these; <c>RecipeContext.TransformerBackends</c> yields both anyway, and one of them lacking a
    /// packed-weight kernel would widen a DiT it never reads.</summary>
    [Fact]
    public void ExcludesTheCfgParallelAndContextParallelPeers()
    {
        using CpuBackend primary = new CpuBackend();
        using CpuBackend cfg = new CpuBackend();
        using CpuBackend rank1 = new CpuBackend();
        RecipeContext context = Context(primary, cfg: cfg, cp: [primary, rank1]);

        Assert.Equal([primary], MiniMaxH3Recipe.ExecutingBackends(context).ToArray());
        Assert.Contains(cfg, context.TransformerBackends);
        Assert.Contains(rank1, context.TransformerBackends);
    }

    /// <summary>A shard backend configured as the primary itself is one device, not two.</summary>
    [Fact]
    public void DoesNotRepeatThePrimaryWhenItIsAlsoTheShardPeer()
    {
        using CpuBackend primary = new CpuBackend();
        Assert.Equal([primary], MiniMaxH3Recipe.ExecutingBackends(Context(primary, shard: primary)).ToArray());
    }
}
