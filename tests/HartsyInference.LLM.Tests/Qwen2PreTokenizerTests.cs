using System.Text.RegularExpressions;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Qwen2 and Qwen3 GGUFs declare <c>tokenizer.ggml.pre = "qwen2"</c>, which selects the single-digit split.</summary>
public sealed class Qwen2PreTokenizerTests
{
    [Theory]
    [InlineData("default", "Default")]
    [InlineData("llama-bpe", "Llama3")]
    [InlineData("gpt-4o", "Gpt4o")]
    [InlineData("qwen2", "Qwen2")]
    [InlineData("kolibri1", "Qwen2")]
    public void PreName_MapsToItsFamily(string pre, string family) =>
        Assert.Equal(family, GgufLanguageModel.PreTokenizerFamilyFor(pre).ToString());

    [Fact]
    public void DefaultFamily_UsesTheGpt2Split() =>
        Assert.Null(GgufLanguageModel.PreTokenRegexFor("default"));

    [Fact]
    public void Qwen2Family_SplitsEveryDigitApart() =>
        Assert.Equal(
            ["year", " ", "2", "0", "2", "4", " and", " ", "3", "6", "5"],
            Regex.Matches("year 2024 and 365", GgufLanguageModel.PreTokenRegexFor("qwen2")!).Select(m => m.Value).ToArray());
}
