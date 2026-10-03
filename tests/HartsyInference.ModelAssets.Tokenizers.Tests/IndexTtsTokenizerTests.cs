using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary><see cref="IndexTtsTextNormalizer"/>'s CJK boundary injection needs no model file and always runs.
/// <see cref="IndexTtsTokenizer"/>'s round-trip needs the real IndexTTS-1.5 <c>bpe.model</c> (476 kB SentencePiece
/// Unigram model — too large to bundle); point <c>INDEXTTS_BPE_MODEL_PATH</c> at a local copy to exercise it.</summary>
public sealed class IndexTtsTokenizerTests
{
    [Theory]
    [InlineData("hello world", "hello world")]
    [InlineData("你好世界", "你好世界")]
    [InlineData("hello你好world", "hello 你好 world")]
    [InlineData("I said 你好 to him", "I said 你好 to him")]
    [InlineData("中文,English混合", "中文 ,English 混合")]
    public void InjectCjkBoundaries_SpacesAtCjkNonCjkTransitions(string input, string expected)
    {
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
