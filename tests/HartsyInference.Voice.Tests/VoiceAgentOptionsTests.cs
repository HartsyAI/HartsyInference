using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary><see cref="VoiceAgentOptions.Validate"/>: the plan's defaults pass, and an option the session cannot honor
/// throws instead of being ignored.</summary>
public sealed class VoiceAgentOptionsTests
{
    [Fact]
    public void TheDefaultsPassValidation()
    {
        new VoiceAgentOptions().Validate();
    }

    [Fact]
    public void PartialTranscriptsAreRejectedNotIgnored()
    {
        VoiceAgentOptions options = new() { PartialTranscripts = true };
        Assert.Throws<NotSupportedException>(options.Validate);
    }

    [Theory]
    [InlineData(nameof(VoiceAgentOptions.EndOfTurnSilenceMs))]
    [InlineData(nameof(VoiceAgentOptions.CpuThreadCap))]
    public void OutOfRangeValuesThrow(string field)
    {
        VoiceAgentOptions options = field switch
        {
            nameof(VoiceAgentOptions.EndOfTurnSilenceMs) => new() { EndOfTurnSilenceMs = 20 },
            _ => new() { CpuThreadCap = -1 },
        };
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
        Assert.Equal(field, error.ParamName);
    }
}
