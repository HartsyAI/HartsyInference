using System.Diagnostics;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Per-stage wall time of the frames a <see cref="HaruDuplexSession"/> computed, kept so the real-time budget can be read per stage
/// instead of guessed: a frame is 133 ms of audio and everything below has to fit in it. Costs one timestamp per stage per frame.</summary>
public sealed class HaruStageTimer
{
    /// <summary>The stages of one frame, in the order they run.</summary>
    public static readonly string[] Stages = ["encode", "embed", "trunk", "logits", "head", "decode"];

    private const int MaxSamples = 8192;
    private readonly List<double>[] _seconds = Array.ConvertAll(Stages, _ => new List<double>());
    private long _mark;

    internal void Begin() => _mark = Stopwatch.GetTimestamp();

    /// <summary>Closes the running stage and starts the next one.</summary>
    internal void Lap(int stage)
    {
        long now = Stopwatch.GetTimestamp();
        List<double> list = _seconds[stage];
        if (list.Count < MaxSamples)
        {
            list.Add(Stopwatch.GetElapsedTime(_mark, now).TotalSeconds);
        }
        _mark = now;
    }

    /// <summary>Median and maximum seconds of a stage over the frames that ran it.</summary>
    public (double Median, double Max, int Frames) Summary(int stage)
    {
        List<double> list = _seconds[stage];
        if (list.Count == 0)
        {
            return (0, 0, 0);
        }
        double[] sorted = [.. list];
        Array.Sort(sorted);
        return (sorted[sorted.Length / 2], sorted[^1], sorted.Length);
    }

    /// <summary>One line: <c>encode 4.1/6 ms, trunk 80.2/95 ms, ...</c> (median/max).</summary>
    public string Describe() => string.Join(", ", Stages.Select((name, i) =>
    {
        (double median, double max, int frames) = Summary(i);
        return frames == 0 ? $"{name} -" : $"{name} {median * 1000:F1}/{max * 1000:F0} ms";
    }));
}
