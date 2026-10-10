using System.Text.RegularExpressions;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Qwen2 and Qwen3 GGUFs declare <c>tokenizer.ggml.pre = "qwen2"</c>, which must select the single-digit split that
/// <c>kolibri1</c> uses. Mapping only <c>kolibri1</c> left Qwen prompts on the GPT-2 split, which tokenized them about 7% longer.</summary>
public sealed class Qwen2PreTokenizerTests
{
    [Theory]
    [InlineData("qwen2")]
    [InlineData("kolibri1")]
    public void QwenNames_SelectTheSingleDigitSplit(string pre) =>
        Assert.Equal(GgufLanguageModel.Qwen2PreTokenRegex, GgufLanguageModel.PreTokenRegexFor(pre));

    [Theory]
    [InlineData("default")]
    [InlineData("llama-bpe")]
    public void OtherNames_KeepTheirOwnSplit(string pre) =>
        Assert.NotEqual(GgufLanguageModel.Qwen2PreTokenRegex, GgufLanguageModel.PreTokenRegexFor(pre));

    [Fact]
    public void QwenText_SplitsEveryDigitApart() =>
        Assert.Equal(
            ["year", " ", "2", "0", "2", "4", " and", " ", "3", "6", "5"],
            Regex.Matches("year 2024 and 365", GgufLanguageModel.Qwen2PreTokenRegex).Select(m => m.Value).ToArray());
}
