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
    private readonly List<double> _frames = [];
    private long _mark;
    private int _gc0 = -1, _gc1, _gc2;

    internal void Begin()
    {
        if (_gc0 < 0)
        {
            (_gc0, _gc1, _gc2) = (GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2));
        }
        _mark = Stopwatch.GetTimestamp();
    }

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

    /// <summary>Records the whole frame, from the user's samples in to the model's audio out.</summary>
    internal void Frame(double seconds)
    {
        if (_frames.Count < MaxSamples)
        {
            _frames.Add(seconds);
        }
    }

    /// <summary>Median, 99th percentile and maximum seconds of a whole frame, and how many frames took longer than <paramref name="budget"/>.</summary>
    public (double Median, double P99, double Max, int Over, int Frames) FrameSummary(double budget)
    {
        if (_frames.Count == 0)
        {
            return (0, 0, 0, 0, 0);
        }
        double[] sorted = [.. _frames];
        Array.Sort(sorted);
        int over = sorted.Count(s => s > budget);
        return (sorted[sorted.Length / 2], sorted[(int)((sorted.Length - 1) * 0.99)], sorted[^1], over, sorted.Length);
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

    /// <summary>One line: <c>encode 4.1/6 ms, trunk 80.2/95 ms, ...</c> (median/max), then the whole frame against <paramref name="budget"/> seconds.</summary>
    public string Describe(double budget)
    {
        string stages = string.Join(", ", Stages.Select((name, i) =>
        {
            (double median, double max, int frames) = Summary(i);
            return frames == 0 ? $"{name} -" : $"{name} {median * 1000:F1}/{max * 1000:F0} ms";
        }));
        (double m, double p99, double max, int over, int n) = FrameSummary(budget);
        if (n == 0)
        {
            return stages;
        }
        string gc = $"GC gen0/1/2 during the run {GC.CollectionCount(0) - _gc0}/{GC.CollectionCount(1) - _gc1}/{GC.CollectionCount(2) - _gc2}";
        return $"{stages}; whole frame median {m * 1000:F0}, p99 {p99 * 1000:F0}, max {max * 1000:F0} ms, {over} of {n} over {budget * 1000:F0} ms; {gc}; slowest frames: {Slowest(5)}";
    }

    /// <summary>The slowest frames with the stage that took the most in each: <c>#412 580 ms (trunk 540)</c>. Frames over budget that share a stage
    /// point at that stage; frames that share only a moment in time point at something else on the machine.</summary>
    public string Slowest(int count)
    {
        int n = Math.Min(_frames.Count, _seconds[Stages.Length - 1].Count);
        return string.Join(", ", Enumerable.Range(0, n).OrderByDescending(i => _frames[i]).Take(count).Select(i =>
        {
            int worst = Enumerable.Range(0, Stages.Length).MaxBy(s => _seconds[s][i]);
            return $"#{i} {_frames[i] * 1000:F0} ms ({Stages[worst]} {_seconds[worst][i] * 1000:F0})";
        }));
    }
}
