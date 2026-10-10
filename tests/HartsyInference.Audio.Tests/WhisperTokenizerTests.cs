using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>WhisperTokenizer surface tests that don't require model files. The full
/// roundtrip test (encode/decode against HF's tokenizer output) needs the downloaded
/// vocab.json + merges.txt — gated behind the network test trait below.</summary>
public sealed class WhisperTokenizerTests
{
    [Fact]
    public void LanguageToTokenId_RoundTrips_EveryLanguage()
    {
        foreach (string lang in WhisperTokenizer.Languages)
        {
            int id = WhisperTokenizer.LanguageToTokenId(lang);
            Assert.Equal(lang, WhisperTokenizer.TokenIdToLanguage(id));
        }
    }

    [Fact]
    public void LanguageToTokenId_Unknown_Throws()
    {
        Assert.Throws<ArgumentException>(() => WhisperTokenizer.LanguageToTokenId("klingon"));
    }

    [Fact]
    public void IsTimestamp_BoundariesAreCorrect()
    {
        Assert.False(WhisperTokenizer.IsTimestamp(WhisperTokenizer.TimestampStartId - 1));
        Assert.True(WhisperTokenizer.IsTimestamp(WhisperTokenizer.TimestampStartId));
        Assert.True(WhisperTokenizer.IsTimestamp(WhisperTokenizer.TimestampStartId + 1500));
        Assert.False(WhisperTokenizer.IsTimestamp(WhisperTokenizer.TimestampStartId + 1501));
    }

}
