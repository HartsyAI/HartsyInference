using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Calls;

/// <summary>One call's voice agent as the host drives it. <see cref="VoiceAgentCallSession"/> is the real one over a
/// <see cref="VoiceAgentSession"/>; the unit tests substitute a scripted one. The contract is the session's: inbound from
/// one thread, outbound from one thread, neither blocks, events raised in order on a pool thread.</summary>
internal interface IVoiceCallSession : IAsyncDisposable
{
    /// <summary>Raised for every session event, in order, on a pool thread.</summary>
    event Action<VoiceAgentEvent>? EventRaised;

    /// <summary>Rate of the audio <see cref="ReadOutbound"/> returns.</summary>
    int OutboundSampleRate { get; }

    Task StartAsync(CancellationToken cancel);

    /// <summary>Caller audio, 16 kHz, ±1. Never blocks.</summary>
    void PushInbound(ReadOnlySpan<float> samples);

    /// <summary>One turn's reply audio at most; <paramref name="turnId"/> is the turn that produced it. Never blocks.</summary>
    int ReadOutbound(Span<float> destination, out int turnId);

    Task SpeakAsync(string text, CancellationToken cancel);

    void PushDtmf(char digit);

    Task EndAsync();
}
