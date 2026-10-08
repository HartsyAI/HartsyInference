namespace HartsyInference.Voice.Audio;

/// <summary>What one 20 ms inbound frame produced in <see cref="VoiceAudioFrontend.ProcessFrame"/>.</summary>
[Flags]
internal enum VoiceFrameEvents
{
    /// <summary>Nothing to act on.</summary>
    None = 0,

    /// <summary>The VAD opened a speech segment.</summary>
    SpeechStarted = 1,

    /// <summary>An utterance ended and should be answered; copy it with <see cref="VoiceAudioFrontend.CopyUtterance"/>.</summary>
    Endpoint = 2,

    /// <summary>An utterance ended over the agent's reply without ever barging in; it is not answered.</summary>
    UtteranceDiscarded = 4,

    /// <summary>The caller spoke over the reply of turn <see cref="VoiceAudioFrontend.BargeInTurn"/> long enough to stop it.</summary>
    BargeIn = 8,

    /// <summary>The DTMF detector queued a key; take it with <see cref="VoiceAudioFrontend.TryTakeDtmf"/>.</summary>
    InbandDtmf = 16,

    /// <summary>The call-progress classifier queued a finding; take it with <see cref="VoiceAudioFrontend.TryTakeCallProgress"/>.</summary>
    CallProgress = 32,
}
