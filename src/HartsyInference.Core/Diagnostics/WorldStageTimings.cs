using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Threading;

namespace HartsyInference.Core.Diagnostics;

/// <summary>The stages of one streamed world run, timed where they happen: the model's chunk work, the consumer's hand-offs, the
/// writer's encode and disk work, and the window's draw.</summary>
public enum WorldStage
{
    /// <summary>Denoising steps and the state write for one chunk: the model's sampling.</summary>
    ChunkSample,

    /// <summary>VAE decode of one chunk, from latents to float pixels.</summary>
    ChunkDecode,

    /// <summary>Conversion of one chunk's float pixels to RGB24 frames.</summary>
    ChunkToRgb,

    /// <summary>The producer waiting for the consumer to take the previous chunk: generation held back by the consumer.</summary>
    ChunkWaitForRoom,

    /// <summary>The consumer waiting for the next frame from the stream: generation is the bottleneck.</summary>
    FrameWaitStream,

    /// <summary>The consumer blocked handing a frame to the spool, which is what the writer's backlog costs the run.</summary>
    FrameSpoolHandoff,

    /// <summary>The consumer blocked handing a frame to the window's buffer, because the window is behind.</summary>
    FramePlaybackHandoff,

    /// <summary>PNG encode of one frame. Summed over the writer threads, so it can exceed the run's wall time.</summary>
    FrameEncode,

    /// <summary>Disk write of one PNG. Summed over the writer threads.</summary>
    FrameWrite,

    /// <summary>The UI thread converting one frame to the window's bitmap and invalidating the picture.</summary>
    WindowDraw,

    /// <summary>Muxing the MP4 from the spooled PNGs at the end of the run.</summary>
    Mux,
}

/// <summary>Wall-clock totals per <see cref="WorldStage"/>, process-wide, for the run in progress. Recording is two interlocked adds
/// and a timestamp read, so it stays on for every run; <see cref="Reset"/> starts a new run's totals.</summary>
/// <remarks>Totals are per event, not per frame: a stage that runs on several threads at once (the writer's encode) reports the sum of
/// their time. Read <see cref="Report"/> as where the time went, not as a critical path.</remarks>
public static class WorldStageTimings
{
    private static readonly int Count = Enum.GetValues<WorldStage>().Length;
    private static readonly long[] Ticks = new long[Count];
    private static readonly long[] Events = new long[Count];

    /// <summary>The current timestamp, to pass to <see cref="Record"/> when the stage ends.</summary>
    public static long Start() => Stopwatch.GetTimestamp();

    /// <summary>Adds the time since <paramref name="start"/> to <paramref name="stage"/>, and counts one event.</summary>
    public static void Record(WorldStage stage, long start)
    {
        long elapsed = Stopwatch.GetTimestamp() - start;
        Interlocked.Add(ref Ticks[(int)stage], elapsed);
        Interlocked.Increment(ref Events[(int)stage]);
    }

    /// <summary>Clears every total, at the start of a run.</summary>
    public static void Reset()
    {
        for (int i = 0; i < Count; i++)
        {
            Interlocked.Exchange(ref Ticks[i], 0);
            Interlocked.Exchange(ref Events[i], 0);
        }
    }

    /// <summary>The totals as a table: stage, events, total seconds and mean milliseconds per event. Stages that never ran are left out.</summary>
    public static string Report()
    {
        StringBuilder text = new();
        text.AppendLine("stage                 events   total_s   mean_ms");
        for (int i = 0; i < Count; i++)
        {
            long events = Interlocked.Read(ref Events[i]);
            if (events == 0)
            {
                continue;
            }
            double totalSeconds = Interlocked.Read(ref Ticks[i]) / (double)Stopwatch.Frequency;
            double meanMs = totalSeconds * 1000.0 / events;
            text.AppendLine(string.Create(CultureInfo.InvariantCulture,
                $"{((WorldStage)i).ToString(),-21} {events,6} {totalSeconds,9:F2} {meanMs,9:F1}"));
        }
        return text.ToString().TrimEnd();
    }
}
