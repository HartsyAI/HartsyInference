using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Stage-level wall-clock and device→host sync attribution for one Whisper transcription, active only under
/// <c>diagnostics.profile</c>. Every mark drains the backend stream first, so a stage's time is its true host + device
/// cost rather than its async launch cost; that serialization is why it stays off by default. Null when profiling is
/// off, so call sites cost one null check.</summary>
internal sealed class WhisperStageTimer
{
    private readonly IBackend _backend;
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly StringBuilder _stages = new();
    private readonly List<(string Name, double Ms, long Syncs, int Count)> _buckets = [];
    private long _syncsAtLap;
    private double _pendingMs;
    private long _pendingSyncs;
    private double _totalMs;
    private long _totalSyncs;
    private float _minMargin = float.PositiveInfinity;

    private WhisperStageTimer(IBackend backend)
    {
        _backend = backend;
        _syncsAtLap = backend.GetD2hSyncCount();
    }

    /// <summary>A timer when <c>diagnostics.profile</c> is on, else null.</summary>
    public static WhisperStageTimer? Start(IBackend backend) => EngineKnobs.Profile.Value ? new WhisperStageTimer(backend) : null;

    /// <summary>Closes the stage named <paramref name="stage"/>, including every <see cref="Accumulate"/> lap since the
    /// previous mark: syncs the device, records the elapsed time and the D2H syncs, restarts the clock.</summary>
    public void Mark(string stage)
    {
        (double ms, long syncs) = Lap();
        ms += _pendingMs;
        syncs += _pendingSyncs;
        _pendingMs = 0;
        _pendingSyncs = 0;
        _totalMs += ms;
        _totalSyncs += syncs;
        if (_stages.Length > 0)
        {
            _stages.Append(' ');
        }
        _stages.Append(stage).Append('=').Append(Format(ms)).Append("ms/").Append(syncs);
    }

    /// <summary>Adds the time since the previous lap to the repeated sub-stage <paramref name="bucket"/> (the parts of a
    /// decode step or an encoder layer); the enclosing <see cref="Mark"/> still counts it.</summary>
    public void Accumulate(string bucket)
    {
        (double ms, long syncs) = Lap();
        _pendingMs += ms;
        _pendingSyncs += syncs;
        for (int i = 0; i < _buckets.Count; i++)
        {
            if (_buckets[i].Name == bucket)
            {
                (string name, double total, long totalSyncs, int count) = _buckets[i];
                _buckets[i] = (name, total + ms, totalSyncs + syncs, count + 1);
                return;
            }
        }
        _buckets.Add((bucket, ms, syncs, 1));
    }

    /// <summary>Records one greedy step's top-1 minus top-2 logit gap; the report carries the smallest.</summary>
    public void NoteMargin(float margin)
    {
        if (margin < _minMargin)
        {
            _minMargin = margin;
        }
    }

    /// <summary>Logs every marked stage, the per-bucket totals and means, and the total under <paramref name="scope"/>.</summary>
    public void Report(string scope)
    {
        StringBuilder line = new();
        line.Append("[Whisper] ").Append(scope).Append(": ").Append(_stages).Append(" total=").Append(Format(_totalMs))
            .Append("ms/").Append(_totalSyncs).Append(" (stage=ms/D2H syncs)");
        foreach ((string name, double ms, long syncs, int count) in _buckets)
        {
            line.Append(" | ").Append(name).Append(": n=").Append(count).Append(' ').Append(Format(ms)).Append("ms (")
                .Append(Format(ms / count)).Append(" each) ").Append(syncs).Append(" syncs");
        }
        if (!float.IsPositiveInfinity(_minMargin))
        {
            line.Append(" | min top1-top2 logit margin=").Append(_minMargin.ToString("G6", CultureInfo.InvariantCulture));
        }
        Logs.Info(line.ToString());
    }

    private (double Ms, long Syncs) Lap()
    {
        _backend.Sync();
        double ms = _stopwatch.Elapsed.TotalMilliseconds;
        _stopwatch.Restart();
        long now = _backend.GetD2hSyncCount();
        long syncs = now - _syncsAtLap;
        _syncsAtLap = now;
        return (ms, syncs);
    }

    private static string Format(double ms) => ms.ToString("F1", CultureInfo.InvariantCulture);
}
