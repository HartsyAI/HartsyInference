using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>The Digits and Punctuation stages of the fish-speech-1.4 <c>tokenizer.json</c>, checked against HuggingFace
/// <c>tokenizers</c> 0.23.2. The expected pieces came from <c>pre_tokenize_str</c> with each byte-level piece decoded back to
/// text, so they are the exact split HF makes before byte-level BPE.</summary>
public sealed class PreTokenizerDigitsPunctuationTests
{
    // The pre_tokenizer block of fishaudio/fish-speech-1.4 tokenizer.json.
    private const string Fish14PreTokenizer =
        """{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"Punctuation","behavior":"Isolated"},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}""";

    private static PreTokenizerPipeline Build()
    {
        using JsonDocument doc = JsonDocument.Parse(Fish14PreTokenizer);
        return PreTokenizerPipeline.FromJson(doc.RootElement.Clone());
    }

    [Theory]
    [InlineData("Hello, world! 123.45", new[] { "Hello", ",", " world", "!", " ", "1", "2", "3", ".", "4", "5" })]
    [InlineData("pi = 3.14159; e=2.718", new[] { "pi", " ", "=", " ", "3", ".", "1", "4", "1", "5", "9", ";", " e", "=", "2", ".", "7", "1", "8" })]
    // HF isolates the ASCII + but not the Unicode symbols ∑ and →; the GPT-2 stage groups those with their neighbours.
    [InlineData("u∑+v a→→b x€€y", new[] { "u", "∑", "+", "v", " a", "→→", "b", " x", "€€", "y" })]
    [InlineData("a_b c`d e\\f", new[] { "a", "_", "b", " c", "`", "d", " e", "\\", "f" })]
    [InlineData("  leading  spaces ", new[] { " ", " leading", " ", " spaces", " " })]
    public void Split_MatchesHuggingFaceFishSpeech14(string text, string[] expected) =>
        Assert.Equal(expected, Build().Split(text));
}
