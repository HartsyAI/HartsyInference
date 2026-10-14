using HartsyInference.Audio.Frontends;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>GPT-SoVITS's English G2P must expand numbers to words before phonemizing. CMUdict has no digit entries,
/// so digits otherwise fall through to the letter-spelling fallback (the cause of "March R.D. at PM" output).</summary>
public sealed class GptSoVitsNumberFrontendTests
{
    [Fact]
    public void G2P_DigitsPhonemizeLikeSpelledOutNumbers()
    {
        // The expected side is the shared normalizer's own spoken form, so this checks the wiring, not the reading.
        string spoken = HartsyInference.Audio.Pipelines.EnglishSpokenForm.Expand("The office opens in 2026 at 8 AM.");
        Assert.Equal(GptSoVitsEnglishG2P.G2P(spoken), GptSoVitsEnglishG2P.G2P("The office opens in 2026 at 8 AM."));
    }

    [Fact]
    public void G2P_NoUnknownPhonesForDigits()
    {
        List<string> phones = GptSoVitsEnglishG2P.G2P("Pay $99.95 on March 3rd.");
        Assert.DoesNotContain("UNK", phones);
    }
}
