using HartsyInference.Audio.Models.CosyVoice;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A reference past CosyVoice 2's reliable length fails loudly with the reason instead of producing garbled or truncated speech.</summary>
public sealed class CosyVoiceReferenceGuardTests
{
    [Fact]
    public void Check_AboveTheLimit_ThrowsWithLengthAndReason()
    {
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => CosyVoiceReferenceGuard.Check(sampleCount: 30 * 24_000, sampleRate: 24_000));
        Assert.Contains("30 s reference", ex.Message, StringComparison.Ordinal);
        Assert.Contains("official implementation", ex.Message, StringComparison.Ordinal);
        Assert.Contains("5 to 15 s", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Check_JustAboveTheLimit_Throws()
    {
        int samples = (int)(CosyVoiceReferenceGuard.MaxSeconds * 16_000) + 1;
        Assert.Throws<InvalidOperationException>(() => CosyVoiceReferenceGuard.Check(samples, sampleRate: 16_000));
    }

    [Fact]
    public void Check_AtTheLimit_Passes()
    {
        CosyVoiceReferenceGuard.Check((int)(CosyVoiceReferenceGuard.MaxSeconds * 24_000), sampleRate: 24_000);
    }

    [Fact]
    public void Check_MeasuresSecondsNotSamples()
    {
        CosyVoiceReferenceGuard.Check(sampleCount: 10 * 44_100, sampleRate: 44_100);
        Assert.Throws<InvalidOperationException>(() => CosyVoiceReferenceGuard.Check(sampleCount: 25 * 16_000, sampleRate: 16_000));
    }

    [Fact]
    public void Check_ShortOrMissingReference_Passes()
    {
        CosyVoiceReferenceGuard.Check(sampleCount: 5 * 24_000, sampleRate: 24_000);
        CosyVoiceReferenceGuard.Check(sampleCount: 0, sampleRate: 0);
        CosyVoiceReferenceGuard.Check(sampleCount: 0, sampleRate: 24_000);
    }
}
