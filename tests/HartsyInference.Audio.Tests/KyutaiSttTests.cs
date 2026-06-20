using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Kyutai;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Kyutai STT: the Helium presets, the DSM 32-codebook Mimi variant, and
/// the shared-table audio-offset math (the net-new index bookkeeping that must be exact).</summary>
public sealed class KyutaiSttTests
{
    [Fact]
    public void Stt2_6B_PresetMatchesHelium()
    {
        KyutaiSttConfig c = KyutaiSttConfig.Stt2_6B;
        Assert.Equal(2_048, c.Helium.HiddenSize);
        Assert.Equal(48, c.Helium.NumHiddenLayers);
        Assert.Equal(32, c.Helium.NumAttentionHeads);
        Assert.Equal(64, c.Helium.HeadDim);              // 2048 / 32
        Assert.False(c.Helium.AttentionBias);
        Assert.Equal(100_000f, c.Helium.RopeTheta);
        Assert.Equal(4_001, c.TextVocab);
        Assert.Equal(32, c.NumCodebooks);
        Assert.Equal(2_049, c.CodebookVocab);
    }

    [Fact]
    public void Stt1B_IsMhaWith128HeadDimAndFrenchVocab()
    {
        KyutaiSttConfig c = KyutaiSttConfig.Stt1B;
        Assert.Equal(16, c.Helium.NumHiddenLayers);
        Assert.Equal(128, c.Helium.HeadDim);             // 2048 / 16
        Assert.Equal(c.Helium.NumAttentionHeads, c.Helium.NumKeyValueHeads);  // MHA, no GQA
        Assert.Equal(8_001, c.TextVocab);
        Assert.Equal(0f, c.AudioSilencePrefixSeconds);
        Assert.Equal(0.5f, c.AudioDelaySeconds);
    }

    [Fact]
    public void AudioOffset_PlacesEachCodebookInDisjointRange()
    {
        KyutaiSttConfig c = KyutaiSttConfig.Stt2_6B;
        Assert.Equal(c.TextVocab, c.AudioOffset(0));                       // codebook 0 starts right after text
        Assert.Equal(c.TextVocab + c.CodebookVocab, c.AudioOffset(1));     // each codebook strides by 2049
        Assert.Equal(c.TextVocab + 31 * c.CodebookVocab, c.AudioOffset(31));
        // The full shared table is text + 32*2049 + 1; the last audio row must fit inside it.
        int sharedRows = c.TextVocab + c.NumCodebooks * c.CodebookVocab + 1;
        Assert.True(c.AudioOffset(31) + c.CodebookVocab <= sharedRows);
    }

    [Fact]
    public void DsmMimi_Has32Codebooks()
    {
        MimiConfig m = MimiConfig.Mimi24kHzDsm;
        Assert.Equal(32, m.TotalCodebooks);              // 1 semantic + 31 acoustic
        Assert.Equal(24_000, m.SampleRate);
    }
}
