using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary><see cref="VoiceAgentOptions.Validate"/>: the plan's defaults pass, and an option the session cannot honor
/// throws instead of being ignored.</summary>
public sealed class VoiceAgentOptionsTests
{
    [Fact]
    public void DefaultsAreThePlansAndValidate()
    {
        VoiceAgentOptions options = new();
        options.Validate();
        Assert.Equal("qwen3", options.LlmModel);
        Assert.Equal("cuda:0", options.LlmDevice);
        Assert.Equal("cuda:1", options.AudioDevice);
        Assert.Equal("whisper:openai/whisper-small.en", options.SttModel);
        Assert.Equal("kokoro:af_heart", options.TtsModel);
        Assert.Equal(16_000, options.OutboundSampleRate);
        Assert.Equal(700, options.EndOfTurnSilenceMs);
        Assert.Equal(15_000, options.MaxUtteranceMs);
        Assert.True(options.BargeInEnabled);
        Assert.Equal(0.6f, options.BargeInProbability);
        Assert.Equal(200, options.BargeInMinMs);
        Assert.Equal(300, options.BargeInHoldoffMs);
        Assert.False(options.Denoise);
        Assert.Equal(4, options.MaxToolRoundsPerTurn);
        Assert.Equal(3_000, options.MaxHistoryTokens);
        Assert.Equal(200, options.MaxReplyTokens);
        Assert.Equal(12, options.FirstSentenceMinChars);
        Assert.Equal(180, options.MaxSentenceChars);
        Assert.Equal(0, options.CpuThreadCap);
        Assert.False(options.PartialTranscripts);
    }

    [Fact]
    public void PartialTranscriptsAreRejectedNotIgnored()
    {
        VoiceAgentOptions options = new() { PartialTranscripts = true };
        Assert.Throws<NotSupportedException>(options.Validate);
    }

    [Theory]
    [InlineData(nameof(VoiceAgentOptions.EndOfTurnSilenceMs))]
    [InlineData(nameof(VoiceAgentOptions.MaxUtteranceMs))]
    [InlineData(nameof(VoiceAgentOptions.BargeInProbability))]
    [InlineData(nameof(VoiceAgentOptions.OutboundSampleRate))]
    [InlineData(nameof(VoiceAgentOptions.MaxToolRoundsPerTurn))]
    [InlineData(nameof(VoiceAgentOptions.CpuThreadCap))]
    public void OutOfRangeValuesThrow(string field)
    {
        VoiceAgentOptions options = field switch
        {
            nameof(VoiceAgentOptions.EndOfTurnSilenceMs) => new() { EndOfTurnSilenceMs = 20 },
            nameof(VoiceAgentOptions.MaxUtteranceMs) => new() { MaxUtteranceMs = 10 },
            nameof(VoiceAgentOptions.BargeInProbability) => new() { BargeInProbability = 0f },
            nameof(VoiceAgentOptions.OutboundSampleRate) => new() { OutboundSampleRate = 0 },
            nameof(VoiceAgentOptions.MaxToolRoundsPerTurn) => new() { MaxToolRoundsPerTurn = 0 },
            _ => new() { CpuThreadCap = -1 },
        };
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
        Assert.Equal(field, error.ParamName);
    }
}
