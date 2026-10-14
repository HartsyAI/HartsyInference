using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Parts of <see cref="IndexTtsFrontNormalizer"/> the upstream fixture does not reach: the interim English
/// normalizer direct pipeline callers get by default, and glossary terms (no released checkpoint ships a glossary).</summary>
public sealed class IndexTtsFrontNormalizerTests
{
    private static readonly IndexTtsFstNormalizers Identity = new() { English = static t => t, Chinese = static t => t };

    [Fact]
    public void InterimNormalizer_ExpandsDatesTimesAndCurrency()
    {
        IndexTtsFrontNormalizer normalizer = new(IndexTtsFstNormalizers.Interim, enableGlossary: false);
        string spoken = normalizer.Normalize("On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.");
        Assert.DoesNotContain(spoken, char.IsDigit);
        Assert.Equal("Yes.", normalizer.Normalize("Yes."));
    }

    [Fact]
    public void Glossary_ReplacesTermsPerLanguage_OnlyWhenEnabled()
    {
        KeyValuePair<string, IndexTtsGlossaryTerm>[] terms =
        [
            new("C++", new IndexTtsGlossaryTerm("C plus plus", "C 加加")),
            new("TTS", new IndexTtsGlossaryTerm(null, "T T S")),
        ];
        IndexTtsFrontNormalizer enabled = new(Identity, enableGlossary: true);
        enabled.LoadGlossary(terms);
        Assert.Equal("I write C plus plus for TTS.", enabled.Normalize("I write c++ for tts."));
        Assert.Equal("我写C 加加的T T S.", enabled.Normalize("我写C++的TTS。"));

        IndexTtsFrontNormalizer disabled = new(Identity, enableGlossary: false);
        disabled.LoadGlossary(terms);
        Assert.Equal("I write c++.", disabled.Normalize("I write c++."));
    }
}
