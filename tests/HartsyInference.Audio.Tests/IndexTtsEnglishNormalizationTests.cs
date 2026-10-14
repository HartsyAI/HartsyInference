using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The IndexTTS v1 and 2.5 text paths must apply the same English spoken-form normalization as 2.0. Digits
/// reaching the tokenizer unexpanded were the cause of garbled numbers (the 2.5 path previously skipped it).</summary>
public sealed class IndexTtsEnglishNormalizationTests
{
    [Fact]
    public void Normalize_ExpandsDatesTimesAndCurrency_ToSpokenWords()
    {
        string spoken = IndexTts2TextNormalizer.Normalize("On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.");
        Assert.DoesNotContain(spoken, c => char.IsDigit(c));
        Assert.Contains("TWENTY", spoken, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Normalize_LeavesPlainWordsUnchanged()
    {
        Assert.Equal("Yes.", IndexTts2TextNormalizer.Normalize("Yes."));
    }
}
