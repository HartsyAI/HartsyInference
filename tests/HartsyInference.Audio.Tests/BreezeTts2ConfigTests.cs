using HartsyInference.Audio.Models.BreezeTts;
using Xunit;

namespace HartsyInference.Audio.Tests;

public sealed class BreezeTts2ConfigTests
{
    [Fact]
    public void Default_MatchesReleasedCheckpointContract()
    {
        BreezeTts2Config config = BreezeTts2Config.Default;

        Assert.Equal(2_048, config.HiddenSize);
        Assert.Equal(28, config.NumBackboneLayers);
        Assert.Equal(6_144, config.BackboneIntermediateSize);
        Assert.Equal(12, config.NumDepthDecoderLayers);
        Assert.Equal(1_024, config.DepthDecoderHiddenSize);
        Assert.Equal(8_192, config.DepthDecoderIntermediateSize);
        Assert.Equal(16, config.NumCodebooks);
        Assert.Equal(32, config.Codec.TotalCodebooks);
        Assert.Equal(2_048, config.Codec.CodebookSize);
        Assert.Equal(1, config.Codec.NumSemanticCodebooks);
        Assert.Equal([1], config.Codec.ResidualDilations);
        Assert.Equal(12.5f, config.CodecFrameRate);
        Assert.Equal(config.CodecFrameRate, (float)config.Codec.FrameRateHz);
        Assert.Equal(24_000, config.Codec.SampleRate);
        Assert.Equal(262_144, config.AudioTokenId);
        Assert.Equal(2_051, config.AudioVocabSize);
    }
}
