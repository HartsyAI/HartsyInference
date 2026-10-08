namespace HartsyInference.Voice;

/// <summary>The kind of a <see cref="VoiceAgentEvent"/>; it names which of the event's payload fields are set.</summary>
public enum VoiceAgentEventKind
{
    /// <summary>The session moved to <see cref="VoiceAgentEvent.State"/>.</summary>
    StateChanged,

    /// <summary>What the caller said, final (<see cref="VoiceAgentEvent.Text"/>).</summary>
    UserTranscript,

    /// <summary>What the agent replied (<see cref="VoiceAgentEvent.Text"/>); after a barge-in, the text generated before it.</summary>
    AssistantTranscript,

    /// <summary>The caller spoke over turn <see cref="VoiceAgentEvent.TurnId"/>; its queued audio is being dropped.</summary>
    BargeIn,

    /// <summary>The model called <see cref="VoiceAgentEvent.ToolCall"/>.</summary>
    ToolCall,

    /// <summary><see cref="VoiceAgentEvent.ToolCall"/> returned <see cref="VoiceAgentEvent.Text"/>.</summary>
    ToolResult,

    /// <summary>Caller audio was not answered (<see cref="VoiceAgentEvent.Text"/> says why): speech over the reply that
    /// never became a barge-in (<see cref="VoiceAgentEvent.TurnId"/> 0, as it never started a turn), or an utterance
    /// the recognizer heard no words in (the id of the turn that transcribed it).</summary>
    UtteranceDiscarded,

    /// <summary>Turn <see cref="VoiceAgentEvent.TurnId"/> ended; <see cref="VoiceAgentEvent.Metrics"/> holds its timings.</summary>
    TurnCompleted,

    /// <summary>Something failed (<see cref="VoiceAgentEvent.Text"/>, <see cref="VoiceAgentEvent.Error"/>); the session
    /// keeps listening unless the audio thread itself failed, which ends it.</summary>
    Error,

    /// <summary>A DTMF tone was heard in the far end's audio (<see cref="VoiceAgentEvent.Dtmf"/>, the key also in
    /// <see cref="VoiceAgentEvent.Text"/>). Only raised when <see cref="VoiceAgentOptions.DetectInbandDtmf"/> is on. This is
    /// audio the session listened to, unlike <see cref="VoiceAgentSession.PushDtmf"/>, where the host reports a key.</summary>
    InbandDtmfDetected,

    /// <summary>The far end's audio looks like ringing, a busy signal, a beep, a recording, a person, music or silence
    /// (<see cref="VoiceAgentEvent.CallProgress"/>, its kind also in <see cref="VoiceAgentEvent.Text"/>). A signal with a
    /// confidence, not a verdict. Only raised when <see cref="VoiceAgentOptions.DetectCallProgress"/> is on.</summary>
    CallProgressDetected,
}
