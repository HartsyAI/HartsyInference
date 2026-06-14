using HartsyInference.Tokenizers;
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
    public void LanguageToTokenId_EnglishIsFirst()
    {
        // English → 50259 (the first language token).
        Assert.Equal(50_259, WhisperTokenizer.LanguageToTokenId("en"));
    }

    [Fact]
    public void LanguageToTokenId_Cantonese_AddedForV3()
    {
        // Cantonese was appended for large-v3 and lives at the end of the language table.
        Assert.Equal(50_358, WhisperTokenizer.LanguageToTokenId("yue"));
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

    [Fact]
    public void TimestampToSeconds_IsExactlyTwoCs_PerStep()
    {
        Assert.Equal(0.0, WhisperTokenizer.TimestampToSeconds(WhisperTokenizer.TimestampStartId), precision: 6);
        Assert.Equal(0.02, WhisperTokenizer.TimestampToSeconds(WhisperTokenizer.TimestampStartId + 1), precision: 6);
        Assert.Equal(30.0, WhisperTokenizer.TimestampToSeconds(WhisperTokenizer.TimestampStartId + 1500), precision: 6);
    }

    [Fact]
    public void Languages_TableHas100Codes_ForLargeV3()
    {
        // Whisper v2 had 99 languages, v3 added Cantonese for a total of 100.
        // Our table is the union so it should hold 100 entries.
        Assert.Equal(100, WhisperTokenizer.Languages.Count);
    }
}
