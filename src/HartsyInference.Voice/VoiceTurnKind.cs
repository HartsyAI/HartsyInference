namespace HartsyInference.Voice;

/// <summary>What started a turn.</summary>
public enum VoiceTurnKind
{
    /// <summary>The caller spoke and the endpoint detector closed the utterance.</summary>
    Utterance,

    /// <summary>The caller pressed a key (<see cref="VoiceAgentSession.PushDtmf"/>).</summary>
    Dtmf,

    /// <summary>The host asked the agent to say something (<see cref="VoiceAgentSession.SpeakAsync"/>).</summary>
    Speak,
}
