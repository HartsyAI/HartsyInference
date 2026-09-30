namespace HartsyInference.Voice;

/// <summary>What a <see cref="VoiceAgentSession"/> is doing.</summary>
public enum VoiceAgentState
{
    /// <summary>Constructed; <see cref="VoiceAgentSession.StartAsync"/> has not run.</summary>
    Created,

    /// <summary>Starting its audio thread and warming the per-frame path.</summary>
    Warming,

    /// <summary>Waiting for the caller to speak.</summary>
    Listening,

    /// <summary>Transcribing the caller or waiting for the language model.</summary>
    Thinking,

    /// <summary>The reply's audio is being produced or played.</summary>
    Speaking,

    /// <summary>A tool the model called is running.</summary>
    ToolRunning,

    /// <summary>Ended; no further audio is accepted.</summary>
    Ended,
}
