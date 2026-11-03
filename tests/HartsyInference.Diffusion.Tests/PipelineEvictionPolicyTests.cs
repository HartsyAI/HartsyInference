using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Which cached pipelines a model switch may evict. The rules are pure, so they run without a device.</summary>
public sealed class PipelineEvictionPolicyTests
{
    [Fact]
    public void FitsBesideCached_WithoutAKnownEstimate_AlwaysFits()
    {
        Assert.True(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 0, estimatedBytes: 0));
    }

    [Fact]
    public void EstimateVariantBytes_FallsBackToTheFileWhenNothingIsMeasured()
    {
        Assert.Equal(1500, PipelineEvictionPolicy.EstimateVariantBytes(measuredResidentBytes: 0, checkpointBytes: 1000));
    }

    [Fact]
    public void EstimateVariantBytes_UsesTheMeasuredResidentSizeWithHeadroom()
    {
        Assert.Equal(2000, PipelineEvictionPolicy.EstimateVariantBytes(measuredResidentBytes: 1600, checkpointBytes: 1000));
    }

    [Fact]
    public void EstimateVariantBytes_RaisesAnIncompleteMeasurementToTheFile()
    {
        Assert.Equal(1250, PipelineEvictionPolicy.EstimateVariantBytes(measuredResidentBytes: 5, checkpointBytes: 1000));
    }

    [Fact]
    public void QuantizedCheckpoint_WhoseResidentFormIsLargerThanItsFile_IsSizedByTheMeasurement()
    {
        // A 1,000-byte file that holds 1,600 bytes on the device. The file rule needs 1,500, so 1,800 free would keep it;
        // the measurement needs 2,000, so the same 1,800 evicts it.
        long fromFile = PipelineEvictionPolicy.EstimateVariantBytes(measuredResidentBytes: 0, checkpointBytes: 1000);
        long measured = PipelineEvictionPolicy.EstimateVariantBytes(measuredResidentBytes: 1600, checkpointBytes: 1000);

        Assert.True(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 1800, estimatedBytes: fromFile));
        Assert.False(PipelineEvictionPolicy.FitsBesideCached(freeBytes: 1800, estimatedBytes: measured));
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
