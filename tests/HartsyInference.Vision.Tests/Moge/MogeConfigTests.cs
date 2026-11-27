using HartsyInference.Vision.Moge;

namespace HartsyInference.Vision.Tests.Moge;

public sealed class MogeConfigTests
{
    private const string Released = """
        {"encoder":"dinov2_vitl14","remap_output":"exp","output_mask":true,"split_head":true,"intermediate_layers":4,
         "dim_upsample":[256,128,64],"dim_times_res_block_hidden":2,"num_res_blocks":2,"trained_area_range":[250000,500000],
         "last_conv_channels":32,"last_conv_size":1}
        """;

    [Fact]
    public void FromJson_ReadsTheReleasedMogeVitLConfig()
    {
        MogeConfig config = MogeConfig.FromJson(Released);

        Assert.Equal(MogeRemap.Exp, config.Remap);
        Assert.Equal(4, config.IntermediateLayers);
        Assert.Equal([256, 128, 64], config.DimUpsample);
        Assert.Equal(2, config.ResBlockHiddenMultiplier);
        Assert.Equal(2, config.NumResBlocks);
        Assert.Equal(512, config.DimProj);
        // The legacy trained_area_range is pixels: divided by 14 squared it is the token range.
        Assert.Equal((250000 / 196, 500000 / 196), config.NumTokensRange);
    }

    [Fact]
    public void FromJson_RejectsAnotherBackboneABooleanRemapAndAMergedHead()
    {
        Assert.Throws<InvalidDataException>(() => MogeConfig.FromJson("""{"encoder":"dinov2_vitb14"}"""));
        Assert.Throws<InvalidDataException>(() => MogeConfig.FromJson("""{"remap_output":true}"""));
        Assert.Throws<InvalidDataException>(() => MogeConfig.FromJson("""{"split_head":false}"""));
        Assert.Throws<InvalidDataException>(() => MogeConfig.FromJson("""{"remap_output":"tanh"}"""));
    }
}
