using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the per-size Whisper config presets match the HuggingFace
/// <c>config.json</c> for each upstream release. Numbers are independently sourced
/// from the OpenAI <c>whisper/model.py</c> ModelDimensions tables — if these drift,
/// it means our preset table has bit-rotted vs upstream.</summary>
public sealed class WhisperConfigTests
{
    [Fact]
    public void Tiny_Has39MParams_FromOpenAI_ModelDimensions()
    {
        WhisperConfig c = WhisperConfig.Tiny;
        Assert.Equal(4, c.EncoderLayers);
        Assert.Equal(4, c.DecoderLayers);
        Assert.Equal(384, c.HiddenSize);
        Assert.Equal(6, c.NumHeads);
        Assert.Equal(1536, c.IntermediateSize);
        Assert.Equal(80, c.NumMelBins);
        Assert.Equal(64, c.HeadDim);
    }

    [Fact]
    public void LargeV3Turbo_HasFourDecoderLayers()
    {
        WhisperConfig c = WhisperConfig.LargeV3Turbo;
        Assert.Equal(32, c.EncoderLayers);
        Assert.Equal(4, c.DecoderLayers);
        Assert.Equal(1280, c.HiddenSize);
        Assert.Equal(20, c.NumHeads);
        Assert.Equal(128, c.NumMelBins);
        Assert.Equal(51_866, c.VocabSize);
    }

    [Fact]
    public void LargeV3_Has128MelBins_AndExtraVocabEntry()
    {
        WhisperConfig c = WhisperConfig.LargeV3;
        Assert.Equal(128, c.NumMelBins);
        Assert.Equal(51_866, c.VocabSize); // +1 vs v2 for Cantonese
        Assert.Equal(32, c.DecoderLayers);
    }

    [Fact]
    public void DistilLargeV3_HasTwoDecoderLayers_ButRestMatchesV3()
    {
        WhisperConfig c = WhisperConfig.DistilLargeV3;
        Assert.Equal(32, c.EncoderLayers);
        Assert.Equal(2, c.DecoderLayers);
        Assert.Equal(128, c.NumMelBins);
        Assert.Equal(1280, c.HiddenSize);
    }

    [Fact]
    public void HeadDim_AlwaysSixtyFour_AcrossAllSizes()
    {
        WhisperConfig[] all = [
            WhisperConfig.Tiny, WhisperConfig.Base, WhisperConfig.Small, WhisperConfig.Medium,
            WhisperConfig.LargeV2, WhisperConfig.LargeV3, WhisperConfig.LargeV3Turbo,
            WhisperConfig.DistilLargeV2, WhisperConfig.DistilLargeV3,
            WhisperConfig.DistilMediumEn, WhisperConfig.DistilSmallEn,
            WhisperConfig.TinyEn, WhisperConfig.BaseEn, WhisperConfig.SmallEn, WhisperConfig.MediumEn,
        ];
        foreach (WhisperConfig cfg in all) Assert.Equal(64, cfg.HeadDim);
    }

    [Fact]
    public void EnglishOnlyPresets_CarryTheEnglishOnlyVocabAndIds()
    {
        // Per the *.en config.json files: vocab 51864, decoder_start 50257, eos/pad 50256, forced [[1, 50362]].
        WhisperConfig[] english = [
            WhisperConfig.TinyEn, WhisperConfig.BaseEn, WhisperConfig.SmallEn, WhisperConfig.MediumEn,
            WhisperConfig.DistilSmallEn, WhisperConfig.DistilMediumEn,
        ];
        foreach (WhisperConfig c in english)
        {
            Assert.False(c.IsMultilingual);
            Assert.Equal(51_864, c.VocabSize);
            Assert.Equal(50_256, c.PadTokenId);
            Assert.Equal(50_256, c.EndOfTextTokenId);
            Assert.Equal(50_257, c.StartOfTranscriptTokenId);
            Assert.Equal(50_258, c.LanguageTokenStart);
            Assert.Equal(50_357, c.TranslateTokenId);
            Assert.Equal(50_358, c.TranscribeTokenId);
            Assert.Equal(50_361, c.NoSpeechTokenId);
            Assert.Equal(50_362, c.NoTimestampsTokenId);
            Assert.Equal(50_363, c.TimestampTokenStart);
        }
        Assert.Equal(WhisperConfig.Small.HiddenSize, WhisperConfig.SmallEn.HiddenSize);
        Assert.Equal(WhisperConfig.Small.EncoderLayers, WhisperConfig.SmallEn.EncoderLayers);
        Assert.Equal(2, WhisperConfig.DistilSmallEn.DecoderLayers);
        Assert.Equal(2, WhisperConfig.DistilMediumEn.DecoderLayers);
    }

    [Fact]
    public void Pipeline_InferConfig_MapsEnglishOnlyReposToEnglishOnlyPresets()
    {
        Assert.Equal(WhisperConfig.TinyEn, WhisperPipeline.InferConfig("openai/whisper-tiny.en"));
        Assert.Equal(WhisperConfig.BaseEn, WhisperPipeline.InferConfig("openai/whisper-base.en"));
        Assert.Equal(WhisperConfig.SmallEn, WhisperPipeline.InferConfig("openai/whisper-small.en"));
        Assert.Equal(WhisperConfig.MediumEn, WhisperPipeline.InferConfig("openai/whisper-medium.en"));
        Assert.Equal(WhisperConfig.DistilSmallEn, WhisperPipeline.InferConfig("distil-whisper/distil-small.en"));
        Assert.NotEqual(WhisperConfig.Small, WhisperPipeline.InferConfig("openai/whisper-small.en"));
    }

    [Fact]
    public void SpecialTokenIds_AreStandard()
    {
        WhisperConfig c = WhisperConfig.Tiny;
        Assert.Equal(50_257, c.EndOfTextTokenId);
        Assert.Equal(50_258, c.StartOfTranscriptTokenId);
        Assert.Equal(50_259, c.LanguageTokenStart);
        Assert.Equal(50_358, c.TranslateTokenId);
        Assert.Equal(50_359, c.TranscribeTokenId);
        Assert.Equal(50_362, c.NoSpeechTokenId);
        Assert.Equal(50_363, c.NoTimestampsTokenId);
        Assert.Equal(50_364, c.TimestampTokenStart);
    }

    [Fact]
    public void Pipeline_InferConfig_MatchesPresets()
    {
        Assert.Equal(WhisperConfig.Tiny, WhisperPipeline.InferConfig("openai/whisper-tiny"));
        Assert.Equal(WhisperConfig.LargeV3Turbo, WhisperPipeline.InferConfig("openai/whisper-large-v3-turbo"));
        Assert.Equal(WhisperConfig.DistilLargeV3, WhisperPipeline.InferConfig("distil-whisper/distil-large-v3"));
        Assert.Throws<ArgumentException>(() => WhisperPipeline.InferConfig("some-random/whisper-fork"));
    }
}
