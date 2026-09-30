namespace HartsyInference.PhoneGateway.Media;

/// <summary>Scheduling and codec settings for <see cref="ClockedAudioSource"/>.</summary>
public sealed record ClockedAudioSourceOptions
{
    /// <summary><c>SCHED_FIFO</c> priority requested for the tick thread; zero never asks.</summary>
    public int FifoPriority { get; init; } = 50;

    /// <summary>CPU the tick thread is pinned to; negative leaves the process affinity alone.</summary>
    public int TickCpu { get; init; } = -1;

    /// <summary>How far before each deadline the non-FIFO fallback stops sleeping and spins.</summary>
    public long SpinTailNs { get; init; } = 150_000;

    /// <summary>Frames emitted back-to-back to absorb a late wake-up; a larger gap resyncs the clock instead.</summary>
    public int MaxCatchUpFrames { get; init; } = 5;

    /// <summary>Unraised ticks the tick thread runs at a 250 µs period before its first real frame.</summary>
    public int WarmUpTicks { get; init; } = 200;

    /// <summary>The laws offered in SDP, in order.</summary>
    public AudioCodecPreference Codec { get; init; } = AudioCodecPreference.Any;

    /// <summary>Name of the tick thread, for `ps -T` and the journal.</summary>
    public string ThreadName { get; init; } = "phone-rtp-tick";
}
