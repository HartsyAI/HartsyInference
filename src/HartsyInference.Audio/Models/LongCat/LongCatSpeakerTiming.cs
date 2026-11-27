namespace HartsyInference.Audio.Models.LongCat;

/// <summary>How the two speakers' speech is laid on the clip's timeline (the demo's <c>audio_type</c>).</summary>
public enum LongCatSpeakerTiming
{
    /// <summary><c>para</c>: both speak from the start; the clip's speech ends with the shorter of the two.</summary>
    Parallel,

    /// <summary><c>add</c>: the second speaker starts when the first one's speech ends.</summary>
    Sequential,
}
