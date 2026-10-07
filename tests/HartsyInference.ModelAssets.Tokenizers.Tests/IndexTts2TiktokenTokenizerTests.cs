using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary><see cref="IndexTts2TiktokenTokenizer"/>'s pure logic — the special-token construction order
/// (<see cref="IndexTts2TiktokenTokenizer.BuildSpecials"/>) and the language-code→embedding-row lookup
/// (<see cref="IndexTts2TiktokenTokenizer.LangToToken"/>) — needs no model file and always runs. The ids this
/// produces are baked into the real checkpoint's trained <c>text_embedding</c>/<c>text_head</c> rows, so any
/// reordering here would silently desync every token id; these tests pin that order down.
/// <para><see cref="IndexTts2TiktokenTokenizer"/>'s own <c>Encode</c>/<c>Decode</c> round-trip needs the real
/// 58,836-rank <c>multilingual_zh_ja_yue_char_del.tiktoken</c> file (too large to bundle a synthetic equivalent
/// of — the loader requires dense, ordered ranks) — covered instead by
/// <c>IndexTts2PipelineRealWeightTests</c>'s real end-to-end synthesis, which exercises this exact tokenizer
/// against the real file.</para></summary>
public sealed class IndexTts2TiktokenTokenizerTests
{
    [Fact]
    public void BuildSpecials_ProducesExactlyTheSpecialCountTheCheckpointExpects()
    {
        const int baseVocabSize = 58_836;
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(baseVocabSize);

        // 58,836 ranks + 1,673 specials = 60,509 == config.yaml's gpt.number_text_tokens exactly.
        Assert.Equal(1_673, specials.Count);
        Assert.Equal(60_509, baseVocabSize + specials.Count);
    }

    [Fact]
    public void BuildSpecials_AssignsIdsDenselyAndInOrder_StartingAtBaseVocabSize()
    {
        const int baseVocabSize = 100;
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(baseVocabSize);

        for (int i = 0; i < specials.Count; i++)
            Assert.Equal(baseVocabSize + i, specials[i].Id);
    }

    [Fact]
    public void BuildSpecials_HasNoDuplicateTokensOrIds()
    {
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(58_836);

        Assert.Equal(specials.Count, specials.Select(s => s.Token).Distinct().Count());
        Assert.Equal(specials.Count, specials.Select(s => s.Id).Distinct().Count());
    }

    [Fact]
    public void BuildSpecials_StartsWithEndOfTextThenStartOfTranscript()
    {
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(0);

        Assert.Equal("<|endoftext|>", specials[0].Token);
        Assert.Equal("<|startoftranscript|>", specials[1].Token);
    }

    [Fact]
    public void BuildSpecials_EndsWith1501WhisperStyleTimestampTokens()
    {
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(0);

        // The timestamp run (i*0.02 for i in [0, 1501)) is the last block BuildSpecials appends.
        List<(string Token, int Id)> tail = specials.Skip(specials.Count - 1_501).ToList();
        Assert.Equal("<|0.00|>", tail[0].Token);
        Assert.Equal("<|30.00|>", tail[^1].Token);
    }

    [Theory]
    [InlineData("en", 0)]
    [InlineData("EN", 0)]
    [InlineData("zh", 1)]
    [InlineData("ja", 7)]
    public void LangToToken_ReturnsTheLanguageCodesDeclarationIndex_CaseInsensitive(string lang, int expectedIndex)
    {
        Assert.Equal(expectedIndex, IndexTts2TiktokenTokenizer.LangToToken(lang));
    }

    [Fact]
    public void LangToToken_FallsBackToCommon_ForAnUnrecognizedCode()
    {
        int commonIndex = IndexTts2TiktokenTokenizer.LangToToken("common");
        Assert.Equal(commonIndex, IndexTts2TiktokenTokenizer.LangToToken("not-a-real-language"));
    }
}
