using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for HeartMuLa: the config maps to a valid CSM-shaped LM (Llama-3B global + Llama-300M depth,
/// 8 codebooks vocab 8197, 48 kHz), and the pipeline constructs cleanly — the LM reuses the verified CSM
/// dual-transformer.</summary>
public sealed class HeartMulaTests
{
    [Fact]
    public void Oss3B_MapsToCsmShapedMusicLm()
    {
        HeartMulaConfig c = HeartMulaConfig.Oss3B;
        Assert.Equal(3_072, c.Lm.Backbone.HiddenSize);
        Assert.Equal(28, c.Lm.Backbone.NumHiddenLayers);   // llama-3B global backbone
        Assert.Equal(3, c.Lm.Decoder.NumHiddenLayers);     // llama-300M depth decoder
        Assert.Equal(8, c.Lm.NumCodebooks);
        Assert.Equal(8_197, c.Lm.AudioVocab);
        Assert.Equal(128_256, c.Lm.TextVocab);
        Assert.Equal(48_000, c.Lm.SampleRate);
        Assert.Equal(512, c.MuqDim);
        Assert.Equal(8_192, c.CodecCodebookSize);
    }

    [Fact]
    public void Pipeline_ConstructsCleanly()
    {
        using HeartMulaPipeline pipe = new(HeartMulaConfig.Oss3B);
        Assert.Equal(48_000, pipe.SampleRate);
    }
}
