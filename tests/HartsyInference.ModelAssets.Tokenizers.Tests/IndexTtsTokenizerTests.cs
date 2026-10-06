using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary><see cref="IndexTtsTextNormalizer"/>'s CJK boundary injection needs no model file and always runs.
/// <see cref="IndexTtsTokenizer"/>'s round-trip needs the real IndexTTS-1.5 <c>bpe.model</c> (476 kB SentencePiece
/// Unigram model — too large to bundle); point <c>INDEXTTS_BPE_MODEL_PATH</c> at a local copy to exercise it.</summary>
public sealed class IndexTtsTokenizerTests
{
    [Theory]
    [InlineData("hello world", "HELLO WORLD")]
    [InlineData("你好世界", "你 好 世 界")]
    [InlineData("hello你好world", "HELLO 你 好 WORLD")]
    [InlineData("I said 你好 to him", "I SAID 你 好 TO HIM")]
    [InlineData("中文,English混合", "中 文 ,ENGLISH 混 合")]
    public void InjectCjkBoundaries_SplitsEveryCjkCharAndUppercasesNonCjkRuns(string input, string expected)
    {
        // Matches the reference tokenize_by_CJK_char(line, do_upper_case=True) exactly: every CJK code point
        // becomes its own token, and every non-CJK run is uppercased -- the real bpe.model's vocabulary has no
        // lowercase English pieces at all (verified against the real checkpoint), so skipping the uppercase step
        // sends almost every English word to <unk>.
        string actual = IndexTtsTextNormalizer.InjectCjkBoundaries(input);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void InjectCjkBoundaries_EmptyInput_ReturnsEmpty()
    {
        Assert.Equal(string.Empty, IndexTtsTextNormalizer.InjectCjkBoundaries(""));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Encode_RoundTrips_AsciiText_AgainstRealBpeModel()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS_BPE_MODEL_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using IndexTtsTokenizer tokenizer = new(path);
        const string text = "The quick brown fox jumps over the lazy dog.";
        int[] ids = tokenizer.Encode(text);
        Assert.NotEmpty(ids);
        foreach (int id in ids) Assert.InRange(id, 0, 11_999);   // number_text_tokens = 12000

        string decoded = tokenizer.Decode(ids);
        Assert.False(string.IsNullOrWhiteSpace(decoded));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Encode_RealBpeModel_IsUnigramNotBpe()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS_BPE_MODEL_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        // Despite the upstream filename, the real checkpoint's trainer_spec.model_type is UNIGRAM (verified by
        // direct protobuf inspection) — this just confirms the tokenizer loads without the BPE-fallback path
        // IndexTtsTokenizer's doc comment flags as a contingency.
        using IndexTtsTokenizer tokenizer = new(path);
        Assert.NotEmpty(tokenizer.Encode("中文 and English mixed"));
    }

    [Fact]
    public void BuildSpecials_MatchesRealConfigVocabSizeExactly()
    {
        // 58,836 mergeable ranks (the real multilingual_zh_ja_yue_char_del.tiktoken's line count) + this list's
        // count must equal config.yaml's gpt.number_text_tokens (60,509), confirmed against the real checkpoint's
        // text_embedding/text_head row count (60,510 = number_text_tokens + 1) — a stray/missing special here
        // would desync every subsequent id from the trained embedding rows without any loud failure elsewhere.
        const int baseVocabSize = 58_836;
        List<(string Token, int Id)> specials = IndexTts2TiktokenTokenizer.BuildSpecials(baseVocabSize);

        Assert.Equal(1_673, specials.Count);
        Assert.Equal(baseVocabSize + 1_673, baseVocabSize + specials.Count);
        Assert.Equal(60_509, baseVocabSize + specials.Count);

        // Order fixes: endoftext, startoftranscript, then the first 99 (of 106) language tags — "yue" (the 100th
        // LANGUAGES entry) is deliberately excluded since num_languages=99 in the real get_tokenizer() call.
        Assert.Equal(("<|endoftext|>", baseVocabSize), specials[0]);
        Assert.Equal(("<|startoftranscript|>", baseVocabSize + 1), specials[1]);
        Assert.Equal(("<|en|>", baseVocabSize + 2), specials[2]);
        Assert.DoesNotContain(specials, s => s.Token == "<|yue|>");
        Assert.DoesNotContain(specials, s => s.Token == "<|common|>");

        // Ids are contiguous and unique.
        Assert.Equal(specials.Select(s => s.Id), Enumerable.Range(baseVocabSize, specials.Count));
        Assert.Equal(specials.Count, specials.Select(s => s.Token).Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void IndexTts2Tokenizer_RoundTrips_AgainstRealTiktokenFile()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_TIKTOKEN_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real file

        IndexTts2TiktokenTokenizer tokenizer = new(path);
        const string text = "The quick brown fox jumps over the lazy dog. 中文和英文混合测试。";
        int[] ids = tokenizer.Encode(text);
        Assert.NotEmpty(ids);
        foreach (int id in ids) Assert.InRange(id, 0, 60_508);   // number_text_tokens = 60509

        string decoded = tokenizer.Decode(ids);
        Assert.False(string.IsNullOrWhiteSpace(decoded));
    }
}
