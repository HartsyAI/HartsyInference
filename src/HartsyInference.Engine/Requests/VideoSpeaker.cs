namespace HartsyInference.Engine.Requests;

/// <summary>One of the two people who may speak.</summary>
public sealed record VideoSpeaker
{
    /// <summary>The person's speech (RIFF/WAVE); null for one who is silent for the whole clip.</summary>
    public AudioClip? Audio { get; init; }

    /// <summary>Where the person is in the init image. Give both speakers' boxes or neither; with neither the first speaker is on the left half
    /// of the picture and the second on the right, each inset by a tenth.</summary>
    public VideoSpeakerBox? Box { get; init; }

    /// <summary>Set by the engine's vocal pre-step: the separated speech as 16 kHz mono. <see cref="Audio"/> stays the recording the soundtrack is
    /// mixed from. Null outside that step.</summary>
    internal float[]? SeparatedSpeech16k { get; init; }
}
