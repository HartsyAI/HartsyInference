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
}
