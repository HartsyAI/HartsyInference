using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2FlowConfigTests
{
    [Fact]
    public void ShippedStructureFlow_ParsesToTheCheckpointArchitecture()
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(Trellis2TestData.ShippedStructureFlowJson);

        Assert.True(config.Dense);
        Assert.Equal(16, config.Resolution);
        Assert.Equal(8, config.InChannels);
        Assert.Equal(1536, config.ModelChannels);
        Assert.Equal(12, config.NumHeads);
        Assert.Equal(128, config.HeadDim);
        Assert.Equal(30, config.NumBlocks);
        // int(1536 * 5.3334) truncates to 8192, the MLP width in the checkpoint.
        Assert.Equal(8192, config.MlpHidden);
        Assert.Equal(4096, config.DenseTokens);
    }

    [Fact]
    public void ShippedShapeFlow_IsSparseWith32Channels()
    {
        Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(Trellis2TestData.ShippedShapeFlow512Json);

        Assert.False(config.Dense);
        Assert.Equal(32, config.Resolution);
        Assert.Equal(32, config.InChannels);
        Assert.Equal(32, config.OutChannels);
        Assert.Equal(1024, config.CondChannels);
    }

    [Fact]
    public void HeadCountDefaultsFromHeadChannelsWhenNumHeadsIsNull()
    {
        string json = Trellis2TestData.ShippedStructureFlowJson.Replace("\"num_heads\": 12", "\"num_heads\": null", StringComparison.Ordinal);

        Assert.Equal(24, Trellis2FlowConfig.FromJson(json).NumHeads);
    }

    [Theory]
    [InlineData("\"pe_mode\": \"rope\"", "\"pe_mode\": \"ape\"")]
    [InlineData("\"share_mod\": true", "\"share_mod\": false")]
    [InlineData("\"SparseStructureFlowModel\"", "\"SomethingElse\"")]
    [InlineData("\"num_heads\": 12", "\"num_heads\": 11")]
    public void UnsupportedOrInconsistentArchitecturesAreRejected(string from, string to)
    {
        string json = Trellis2TestData.ShippedStructureFlowJson.Replace(from, to, StringComparison.Ordinal);

        Assert.ThrowsAny<Exception>(() => Trellis2FlowConfig.FromJson(json));
    }

    [Fact]
    public void ARopeFrequencyOtherThanTheDefaultIsRejected()
    {
        string json = Trellis2TestData.ShippedStructureFlowJson.Replace(
            "\"qk_rms_norm\"", "\"rope_freq\": [1.0, 500.0], \"qk_rms_norm\"", StringComparison.Ordinal);

        Assert.Throws<NotSupportedException>(() => Trellis2FlowConfig.FromJson(json));
    }
}
