using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class ShapeVaeConfigTests
{
    [Fact]
    public void ShippedConfig_Validates()
    {
        ShapeVaeConfig shipped = ShapeVaeConfig.NextDcF16C32;

        shipped.Validate();
        Assert.Equal(new[] { 64, 128, 256, 512, 1024 }, shipped.Encoder.ModelChannels);
        Assert.Equal(new[] { 1024, 512, 256, 128, 64 }, shipped.Decoder.ModelChannels);
        Assert.Equal(256, shipped.Decoder.Resolution);
    }

    [Fact]
    public void SyntheticConfigs_Validate()
    {
        ShapeVaeTestData.Encoder().Validate();
        ShapeVaeTestData.Decoder().Validate();
    }

    [Fact]
    public void Encoder_RejectsBlockCountLengthMismatch()
    {
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder() with { NumBlocks = [1, 1] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Encoder_RejectsUnsupportedBlockType()
    {
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder() with
        {
            BlockType = ["SparseConvNeXtBlock3d", "SparseResBlock3d", "SparseConvNeXtBlock3d"],
        };
        ArgumentException ex = Assert.Throws<ArgumentException>(() => config.Validate());
        Assert.Contains("SparseResBlock3d", ex.Message);
    }

    [Fact]
    public void Encoder_RejectsUnsupportedDownsampleType()
    {
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder() with { DownBlockType = ["SparseResBlockDownsample3d", "SparseResBlockS2C3d"] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Encoder_RejectsTransitionWhoseOutputIsNotMultipleOfEight()
    {
        // 16 -> 12: the conv1 output (12/8) is not integral.
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder() with { ModelChannels = [8, 16, 12] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Encoder_RejectsSkipThatCannotSplitIntoGroups()
    {
        // 8*24 = 192 is not divisible by 128.
        ShapeVaeEncoderConfig config = ShapeVaeTestData.Encoder() with { ModelChannels = [24, 128, 256] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Encoder_RejectsNonPositiveChannelsAndNegativeBlocks()
    {
        Assert.Throws<ArgumentException>(() => (ShapeVaeTestData.Encoder() with { ModelChannels = [8, 0, 32] }).Validate());
        Assert.Throws<ArgumentException>(() => (ShapeVaeTestData.Encoder() with { NumBlocks = [1, -1, 2] }).Validate());
    }

    [Fact]
    public void Encoder_RejectsNonPositiveLatentChannels()
    {
        Assert.Throws<ArgumentException>(() => (ShapeVaeTestData.Encoder() with { LatentChannels = 0 }).Validate());
    }

    [Fact]
    public void Decoder_RejectsNonPositiveResolution()
    {
        Assert.Throws<ArgumentException>(() => (ShapeVaeTestData.Decoder() with { Resolution = 0 }).Validate());
    }

    [Fact]
    public void Decoder_RejectsInputChannelsNotMultipleOfEight()
    {
        // 12 channels cannot be split into 8 children.
        ShapeVaeDecoderConfig config = ShapeVaeTestData.Decoder() with { ModelChannels = [12, 16, 8] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Decoder_RejectsOutputNotDivisibleByChildChannels()
    {
        // cin 32 gives 4 channels per child; cout 6 does not split into 4.
        ShapeVaeDecoderConfig config = ShapeVaeTestData.Decoder() with { ModelChannels = [32, 6, 8] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Decoder_RejectsUnsupportedUpsampleType()
    {
        ShapeVaeDecoderConfig config = ShapeVaeTestData.Decoder() with { UpBlockType = ["SparseResBlockUpsample3d", "SparseResBlockC2S3d"] };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Constructors_ValidateConfigBeforeLoadingWeights()
    {
        ShapeVaeEncoderConfig badEncoder = ShapeVaeTestData.Encoder() with { ModelChannels = [8, 16, 12] };
        ShapeVaeDecoderConfig badDecoder = ShapeVaeTestData.Decoder() with { Resolution = 0 };

        // An empty weight map would fail with HartsyInferenceException; config validation must run first.
        Assert.Throws<ArgumentException>(() => new ShapeVaeEncoder(badEncoder, new Dictionary<string, HartsyInference.Core.Tensors.Tensor>()));
        Assert.Throws<ArgumentException>(() => new ShapeVaeDecoder(badDecoder, new Dictionary<string, HartsyInference.Core.Tensors.Tensor>()));
    }
}
