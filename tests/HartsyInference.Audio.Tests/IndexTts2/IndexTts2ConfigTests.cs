using HartsyInference.Audio.Models.IndexTts2;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Pins the two version presets against the values in the real <c>config.yaml</c> files: 2.0 and 2.5 are
/// the same stack except the text vocabulary, the semantic codec's resample scale and the version tag itself.</summary>
public sealed class IndexTts2ConfigTests
{
    [Fact]
    public void V2_5_IsUnchanged()
    {
        IndexTts2Config c = IndexTts2Config.V2_5;
        Assert.Equal(IndexTts2Version.V2_5, c.Version);
        Assert.Equal(60_509, c.NumberTextTokens);
        Assert.Equal(2, c.SemanticCodec.DownsampleScale);
    }

    [Fact]
    public void V2_0_MatchesTheRealConfigYaml()
    {
        IndexTts2Config c = IndexTts2Config.V2_0;
        Assert.Equal(IndexTts2Version.V2_0, c.Version);
        Assert.Equal(12_000, c.NumberTextTokens);   // gpt.number_text_tokens; the checkpoint's embedding has 12001 rows
        Assert.Equal(1, c.SemanticCodec.DownsampleScale);   // MaskGCT RepCodec: no resample
        Assert.Equal(8192, c.SemanticCodec.CodebookSize);
    }

    [Fact]
    public void V2_0_SharesEveryOtherFieldWithV2_5()
    {
        IndexTts2Config a = IndexTts2Config.V2_0, b = IndexTts2Config.V2_5;
        Assert.Equal(b.Gpt, a.Gpt);
        Assert.Equal(b.MaxTextTokens, a.MaxTextTokens);
        Assert.Equal(b.MaxMelTokens, a.MaxMelTokens);
        Assert.Equal(b.S2MelDit, a.S2MelDit);
        Assert.Equal(b.BigVgan.UpsampleRates, a.BigVgan.UpsampleRates);
        Assert.Equal(b.BigVgan.UpsampleKernelSizes, a.BigVgan.UpsampleKernelSizes);
        Assert.Equal(b.BigVgan.UpsampleInitialChannel, a.BigVgan.UpsampleInitialChannel);
        Assert.Equal(b.SampleRate, a.SampleRate);
        Assert.Equal(b.ContentLengthRatio, a.ContentLengthRatio);
        Assert.Equal(b.DiffusionSteps, a.DiffusionSteps);
        Assert.Equal(b.InferenceCfgRate, a.InferenceCfgRate);
    }
}
