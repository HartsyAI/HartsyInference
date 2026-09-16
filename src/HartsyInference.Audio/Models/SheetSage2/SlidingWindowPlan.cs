namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>One pass of the transcriber over a five-minute window of audio.</summary>
/// <param name="Start">Where the window begins, in seconds.</param>
/// <param name="End">Where it ends — clamped to the clip, so the last window may be short.</param>
/// <param name="AcceptStart">First second whose events this window is trusted for.</param>
/// <param name="AcceptEnd">One past the last second this window is trusted for.</param>
/// <param name="PrefixEnd">Events before this are replayed as context so the window continues the previous
/// one's reading rather than starting over.</param>
/// <param name="GenerationStop">Seconds into the window after which decoding may stop, since everything past
/// it belongs to the next window; null on the last window, which runs to the end.</param>
public readonly record struct SheetSage2Window(
    double Start, double End, double AcceptStart, double AcceptEnd, double PrefixEnd, double? GenerationStop);

/// <summary>How a clip longer than the encoder's window is divided up.
///
/// <para>The encoder attends over a fixed five minutes, so a longer song is read in overlapping passes. Each
/// window is only trusted for the part of itself that is far enough from its own end — the last a hundred
/// seconds are lookahead, kept for context but discarded, because a transcription near the edge of a window has
/// no idea what follows. Successive windows then hop by the part that was kept.</para></summary>
public static class SlidingWindowPlan
{
    /// <summary>Seconds of audio the encoder reads at once.</summary>
    public const double WindowSeconds = 300.0;

    /// <summary>Seconds each window shares with the previous one.</summary>
    public const double OverlapSeconds = 200.0;

    /// <summary>Seconds at the end of a window that are read for context but not trusted.</summary>
    public const double LookaheadSeconds = 100.0;

    /// <summary>Divides a clip into windows whose accepted spans tile it exactly once.</summary>
    /// <param name="durationSeconds">Length of the clip.</param>
    public static IReadOnlyList<SheetSage2Window> For(double durationSeconds)
    {
        if (!double.IsFinite(durationSeconds) || durationSeconds <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(durationSeconds),
                durationSeconds, "A clip must have a positive, finite duration to transcribe.");
        }
        const double hop = WindowSeconds - OverlapSeconds;
        List<SheetSage2Window> windows = [];
        double start = 0.0;
        double accepted = 0.0;
        while (true)
        {
            bool last = start + WindowSeconds >= durationSeconds - 1e-6;
            double acceptEnd = last ? durationSeconds : start + WindowSeconds - LookaheadSeconds;
            windows.Add(new SheetSage2Window(
                Start: start,
                End: Math.Min(durationSeconds, start + WindowSeconds),
                AcceptStart: accepted,
                AcceptEnd: acceptEnd,
                PrefixEnd: accepted,
                GenerationStop: last ? null : WindowSeconds - LookaheadSeconds));
            if (last) return windows;
            accepted = acceptEnd;
            // The final window is pulled back so it still reads a full window's worth of audio.
            start = Math.Min(start + hop, durationSeconds - WindowSeconds);
        }
    }
}
