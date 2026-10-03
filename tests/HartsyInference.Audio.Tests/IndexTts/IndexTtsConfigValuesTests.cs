using HartsyInference.Audio.Models.IndexTts;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts;

/// <summary>Locks the hand-ported <c>IndexTtsConfig.V1_5</c> preset's numeric constants to the real, downloaded
/// <c>IndexTeam/IndexTTS-1.5/config.yaml</c> — a typo here silently breaks every shape in the pipeline. This is a
/// plain value-equality test, not a <c>SyntheticSmoke</c> run of the pipeline itself (that is deferred; see
/// <c>docs/Research/INDEX_TTS_ARCHITECTURE.md</c>'s implementation-notes addendum).</summary>
public sealed class IndexTtsConfigValuesTests
{
    [Fact]
    public void Gpt_MatchesRealConfigYaml()
    {
        var gpt = IndexTtsConfig.V1_5.Gpt;
        Assert.Equal(1_280, gpt.Hidden);
        Assert.Equal(24, gpt.NumLayers);
        Assert.Equal(20, gpt.NumHeads);
        Assert.Equal(1_402, gpt.BlockSize);   // max_mel_tokens(800) + max_text_tokens(600) + 2
    }

    [Fact]
    public void ConditioningEncoder_MatchesRealConfigYaml()
    {
        var c = IndexTtsConfig.V1_5.ConditioningEncoder;
        Assert.Equal(100, c.InputSize);
        Assert.Equal(512, c.OutputSize);
        Assert.Equal(8, c.AttentionHeads);
        Assert.Equal(2_048, c.LinearUnits);
        Assert.Equal(6, c.NumBlocks);
    }

    [Fact]
    public void BigVgan_MatchesRealConfigYaml()
    {
        var b = IndexTtsConfig.V1_5.BigVgan;
        Assert.Equal(1_280, b.GptDim);
        Assert.Equal(512, b.SpeakerEmbeddingDim);
        Assert.Equal(1_536, b.UpsampleInitialChannel);
        Assert.Equal(new[] { 4, 4, 4, 4, 2, 2 }, b.UpsampleRates);
        Assert.Equal(new[] { 8, 8, 4, 4, 4, 4 }, b.UpsampleKernelSizes);
        Assert.Equal(new[] { 3, 7, 11 }, b.ResblockKernelSizes);
    }

    [Fact]
    public void TopLevel_MatchesRealConfigYaml()
    {
        var cfg = IndexTtsConfig.V1_5;
        Assert.Equal(600, cfg.MaxTextTokens);
        Assert.Equal(800, cfg.MaxMelTokens);
        Assert.Equal(24_000, cfg.SampleRate);
    }

    [Fact]
    public void MelCodeSentinels_MatchRealConfigYaml()
    {
        Assert.Equal(8_192, HartsyInference.Audio.Models.IndexTts.IndexTtsT2sDecoder.StartMelToken);
        Assert.Equal(8_193, HartsyInference.Audio.Models.IndexTts.IndexTtsT2sDecoder.StopMelToken);
        Assert.Equal(8_194, HartsyInference.Audio.Models.IndexTts.IndexTtsT2sDecoder.NumMelCodes);
        Assert.Equal(12_001, HartsyInference.Audio.Models.IndexTts.IndexTtsT2sDecoder.NumTextTokens);
    }
}
