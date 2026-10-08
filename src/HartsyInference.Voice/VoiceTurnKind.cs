namespace HartsyInference.Voice;

/// <summary>What started a turn.</summary>
public enum VoiceTurnKind
{
    /// <summary>The caller spoke and the endpoint detector closed the utterance.</summary>
    Utterance,

    /// <summary>A key: one the host reported (<see cref="VoiceAgentSession.PushDtmf"/>) or, when enabled, one heard in the
    /// audio (<see cref="VoiceAgentOptions.ForwardInbandDtmfToModel"/>).</summary>
    Dtmf,

    /// <summary>The host asked the agent to say something (<see cref="VoiceAgentSession.SpeakAsync"/>).</summary>
    Speak,
}
