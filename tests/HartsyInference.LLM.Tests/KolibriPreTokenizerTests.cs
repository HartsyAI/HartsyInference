using System.Text.RegularExpressions;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Kolibri-1's tokenizer.json splits digits one at a time and treats contractions case-insensitively; the
/// GPT-2 default groups digit runs, which changes the token ids of any prompt containing numbers.</summary>
public sealed class KolibriPreTokenizerTests
{
    private static string[] Split(string text) =>
        Regex.Matches(text, GgufLanguageModel.Qwen2PreTokenRegex).Select(m => m.Value).ToArray();

    [Fact]
    public void Digits_AreSplitOneAtATime() =>
        Assert.Equal(["Year", " ", "2", "0", "2", "6"], Split("Year 2026"));

    [Fact]
    public void Contractions_AreCaseInsensitive() =>
        Assert.Equal(["I", "'M", " here"], Split("I'M here"));

    [Fact]
    public void NewlineRuns_StayTogether() =>
        Assert.Equal(["a", "\n\n", "b"], Split("a\n\nb"));
}
