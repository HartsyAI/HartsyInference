using HartsyInference.Tools;

namespace HartsyInference.VoiceHost.Calls;

/// <summary>Creates a call's session with the tools bound to that call.</summary>
internal interface IVoiceCallSessionFactory
{
    /// <summary>Rate of every session's outbound audio; announced to the gateway in <c>HelloAck</c>.</summary>
    int OutboundSampleRate { get; }

    IVoiceCallSession Create(uint callId, ToolRegistry tools);
}
