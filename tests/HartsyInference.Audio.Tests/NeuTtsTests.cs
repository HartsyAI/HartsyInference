using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.NeuTts;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for NeuTTS Air + NeuCodec: the Qwen2.5-0.5B preset with extended vocab, the
/// speech-token offset, and the single FSQ codebook geometry.</summary>
public sealed class NeuTtsTests
{
    [Fact]
    public void Air_PresetIsQwen25_0_5B_WithExtendedVocab()
    {
        NeuTtsConfig c = NeuTtsConfig.Air;
        Assert.Equal(896, c.Llm.HiddenSize);
        Assert.Equal(24, c.Llm.NumHiddenLayers);
        Assert.Equal(14, c.Llm.NumAttentionHeads);
        Assert.Equal(2, c.Llm.NumKeyValueHeads);          // GQA 7:1
        Assert.Equal(1_000_000f, c.Llm.RopeTheta);
        Assert.True(c.Llm.TieWordEmbeddings);
        Assert.Equal(217_652, c.Llm.VocabSize);           // extended for 65536 speech tokens
        Assert.Equal(151_671, c.SpeechTokenBase);
        Assert.Equal(65_536, c.CodebookSize);
    }

    [Fact]
    public void SpeechTokenRange_FitsInsideVocab()
    {
        NeuTtsConfig c = NeuTtsConfig.Air;
        // The last speech token id must be addressable in the LM vocab.
        Assert.True(c.SpeechTokenBase + c.CodebookSize - 1 < c.Llm.VocabSize);
        Assert.Equal(217_206, c.SpeechTokenBase + c.CodebookSize - 1);   // <|speech_65535|>
    }

    [Fact]
    public void NeuCodec_SingleFsqCodebookIs65536()
    {
        NeuCodecConfig cfg = NeuCodecConfig.Default;
        Assert.Equal(8, cfg.FsqLevels.Count);
        Assert.Equal(65_536, Fsq.VocabSize([.. cfg.FsqLevels]));   // 4^8
        Assert.Equal(65_536, cfg.CodebookSize);
        Assert.Equal(64, cfg.HeadDim);                              // 1024 / 16
        Assert.Equal(1_920, cfg.NFft);                             // hop 480 × 4
    }

    [Fact]
    public void NeuCodecDecoder_ConstructsCleanly()
    {
        NeuCodecDecoder d = new(NeuCodecConfig.Default);
        Assert.Equal(24_000, d.SampleRate);
        Assert.Equal(50, d.FrameRate);
        Assert.Equal(1, d.NCodebooks);
    }
}
