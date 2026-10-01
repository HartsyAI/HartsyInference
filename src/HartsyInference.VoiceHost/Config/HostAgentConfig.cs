using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Config;

/// <summary>Conversation settings for every call (<c>agent</c> section); the defaults are
/// <see cref="VoiceAgentOptions"/>'s.</summary>
public sealed record HostAgentConfig
{
    private static readonly VoiceAgentOptions _defaults = new();

    /// <summary>Instruction every conversation starts with; null means <see cref="VoiceAgentOptions.DefaultSystemPrompt"/>.</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>Said when a call starts, before the caller speaks; null or empty says nothing.</summary>
    public string? Greeting { get; set; }

    /// <summary>Said on a call the gateway re-attaches after the link dropped (a host restart); empty says nothing.</summary>
    public string ResumeApology { get; set; } = "Sorry, the line dropped for a moment. Could you say that again?";

    /// <summary>Rate of the audio sent to the gateway; one of the PhoneLink outbound rates.</summary>
    public int OutboundSampleRate { get; set; } = _defaults.OutboundSampleRate;

    public int EndOfTurnSilenceMs { get; set; } = _defaults.EndOfTurnSilenceMs;

    public int MaxUtteranceMs { get; set; } = _defaults.MaxUtteranceMs;

    public bool BargeInEnabled { get; set; } = _defaults.BargeInEnabled;

    public float BargeInProbability { get; set; } = _defaults.BargeInProbability;

    public int BargeInMinMs { get; set; } = _defaults.BargeInMinMs;

    public int BargeInHoldoffMs { get; set; } = _defaults.BargeInHoldoffMs;

    public int MaxToolRoundsPerTurn { get; set; } = _defaults.MaxToolRoundsPerTurn;

    public int MaxHistoryTokens { get; set; } = _defaults.MaxHistoryTokens;

    public int MaxReplyTokens { get; set; } = _defaults.MaxReplyTokens;

    public int FirstSentenceMinChars { get; set; } = _defaults.FirstSentenceMinChars;

    public int MaxSentenceChars { get; set; } = _defaults.MaxSentenceChars;
}
