using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests.Stt;

/// <summary>Per-model window limits derived from each model's config; no weights are loaded.</summary>
public sealed class SpeechWindowLimitTests
{
    [Fact]
    public void Whisper_WindowIsThirtySeconds()
    {
        Assert.Equal(30, WhisperPipeline.WindowSeconds);
    }

    [Theory]
    [InlineData(5.0, 32)]
    [InlineData(29.0, 188)]
    [InlineData(60.0, 193)]
    public void Moonshine_DerivedBudgetFloorsScalesAndClamps(double seconds, int expected)
    {
        Assert.Equal(expected, MoonshineGreedyDecoder.DerivedTokenBudget(seconds, maxTextPositions: 194));
    }

    [Fact]
    public void Moonshine_StreamingBudgetUsesItsLargerPositionCount()
    {
        Assert.Equal(390, MoonshineGreedyDecoder.DerivedTokenBudget(60.0, maxTextPositions: 448));
        Assert.Equal(447, MoonshineGreedyDecoder.DerivedTokenBudget(120.0, maxTextPositions: 448));
    }

    [Fact]
    public void Moonshine_WindowFollowsDecoderPositions()
    {
        Assert.Equal(193 / 6.5, MoonshineGreedyDecoder.WindowSecondsFor(194), precision: 9);
        Assert.Equal(447 / 6.5, MoonshineGreedyDecoder.WindowSecondsFor(448), precision: 9);
    }

    [Fact]
    public void Kyutai_OneBWindowFitsItsPositionsLessBosAndPadding()
    {
        // 375 positions, 12.5 Hz frames, 0.5 s delay and no silence prefix, one frame reserved for padding.
        Assert.Equal(29.26, KyutaiSttConfig.Stt1B.MaxInputSeconds, precision: 2);
    }

    [Fact]
    public void Kyutai_TwoPointSixBWindowFitsItsPositionsLessBosAndPadding()
    {
        // 750 positions, 12.5 Hz frames, 2.5 s delay and a 1.0 s silence prefix, one frame reserved for padding.
        Assert.Equal(56.26, KyutaiSttConfig.Stt2_6B.MaxInputSeconds, precision: 2);
    }
}
