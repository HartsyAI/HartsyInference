using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The pre-tokenizer embedded for <c>tokenizer.ggml.pre = flint</c> splits like the checkpoint's own: the pieces are those that
/// Hugging Face tokenizers' <c>pre_tokenizer.pre_tokenize_str</c> returns for the same text, decoded from byte-level form back to text (tokenizers 0.23, RecursiveSI/flint <c>tokenizer.json</c>). No model files needed.</summary>
public sealed class FlintPreTokenizerTests
{
    public static TheoryData<string, string[]> Samples() => new()
    {
        { "Hello, world! 12345 + 678 = 813.",
            ["Hello", ",", " world", "!", " ", "123", "45", " +", " ", "678", " =", " ", "813", "."] },
        { "def f(x):\n    return x * 2\n",
            ["def", " f", "(x", "):\n", "   ", " return", " x", " *", " ", "2", "\n"] },
        { "\u4e2d\u6587\u6d4b\u8bd5\uff1a\u4f60\u597d\uff0c\u4e16\u754c\u3002Multiple   spaces\tand tabs.",
            ["\u4e2d\u6587\u6d4b\u8bd5", "\uff1a", "\u4f60\u597d", "\uff0c", "\u4e16\u754c", "\u3002Multiple", "  ", " spaces", "\tand", " tabs", "."] },
        { "CamelCaseWords don't stop, it's 100% fine",
            ["Camel", "Case", "Words", " don't", " stop", ",", " it's", " ", "100", "%", " fine"] },
    };

    [Theory]
    [MemberData(nameof(Samples))]
    public void SplitsLikeTheCheckpointsOwnPreTokenizer(string text, string[] expected) =>
        Assert.Equal(expected, GgufLanguageModel.PreTokenizerPipelineFor("flint")!.Split(text));

    [Fact]
    public void OtherPreTokenizerNamesKeepTheirRegexPath() => Assert.Null(GgufLanguageModel.PreTokenizerPipelineFor("llama-bpe"));
}
