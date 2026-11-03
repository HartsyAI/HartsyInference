using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Which cached pipelines a model switch may evict. The rules are pure, so they run without a device.</summary>
public sealed class PipelineEvictionPolicyTests
{
    [Fact]
    public void FitsBesideCached_WithoutAKnownCheckpointSize_AlwaysFits()
    {
        Assert.True(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 0, checkpointBytes: 0));
    }

    [Fact]
    public void FitsBesideCached_NeedsTheResidentFactorOverTheCheckpointSize()
    {
        Assert.True(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 1500, checkpointBytes: 1000));
        Assert.False(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 1499, checkpointBytes: 1000));
    }

    [Fact]
    public void SelectVictims_KeepsSameCheckpointVariantsWhileMemoryFits()
    {
        string[] keys = { "recipe:/a.safetensors|x", "recipe:/a.safetensors|y", "recipe:/b.safetensors|z" };

        List<string> victims = PipelineEvictionPolicy.SelectVictims(keys, "recipe:/a.safetensors|", null, evictSameCheckpoint: false);

        Assert.Equal(new[] { "recipe:/b.safetensors|z" }, victims);
    }

    [Fact]
    public void SelectVictims_EvictsSameCheckpointVariantsWhenMemoryIsTight()
    {
        string[] keys = { "recipe:/a.safetensors|x", "recipe:/a.safetensors|y", "recipe:/b.safetensors|z" };

        List<string> victims = PipelineEvictionPolicy.SelectVictims(keys, "recipe:/a.safetensors|", null, evictSameCheckpoint: true);

        Assert.Equal(keys, victims);
    }

    [Fact]
    public void SelectVictims_NeverEvictsTheAlsoKeepSurvivor()
    {
        string[] keys = { "recipe:/a.safetensors|x", "recipe:/refiner.safetensors|r", "recipe:/b.safetensors|z" };

        List<string> victims = PipelineEvictionPolicy.SelectVictims(
            keys, "recipe:/a.safetensors|", "recipe:/refiner.safetensors|", evictSameCheckpoint: true);

        Assert.Equal(new[] { "recipe:/a.safetensors|x", "recipe:/b.safetensors|z" }, victims);
    }
}
