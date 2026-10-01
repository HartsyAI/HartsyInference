using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Calls;

/// <summary>The real call session: a <see cref="VoiceAgentSession"/> on the host's model set.</summary>
internal sealed class VoiceAgentCallSession(VoiceAgentSession session) : IVoiceCallSession
{
    public event Action<VoiceAgentEvent>? EventRaised
    {
        add => session.EventRaised += value;
        remove => session.EventRaised -= value;
    }

    public int OutboundSampleRate => session.OutboundSampleRate;

    public Task StartAsync(CancellationToken cancel) => session.StartAsync(cancel);

    public void PushInbound(ReadOnlySpan<float> samples) => session.PushInbound(samples);

    public int ReadOutbound(Span<float> destination, out int turnId) => session.ReadOutbound(destination, out turnId);

    public Task SpeakAsync(string text, CancellationToken cancel) => session.SpeakAsync(text, cancel);

    public void PushDtmf(char digit) => session.PushDtmf(digit);

    public Task EndAsync() => session.EndAsync();

    public ValueTask DisposeAsync() => session.DisposeAsync();
}
