using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Diagnostics;

/// <summary>Stage-level wall-clock and device→host sync attribution for one model call, active only under
/// <c>diagnostics.profile</c>. Every mark drains the backend stream first, so a stage's time is its true host + device
/// cost rather than its async launch cost; that serialization is why it stays off by default. Null when profiling is
/// off, so call sites cost one null check.</summary>
/// <remarks>The per-op profile only sees backend ops; host loops between them (a scalar GEMV, a reshape through
/// <c>DataPointer</c>, an FFT) show up only here, as a stage whose time the op table cannot account for.</remarks>
internal sealed class StageTimer
{
    private readonly IBackend _backend;
    private readonly string _model;
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly StringBuilder _stages = new();
    private readonly List<(string Name, double Ms, long Syncs, int Count)> _buckets = [];
    private readonly List<(string Name, float Value)> _minimums = [];
    private long _syncsAtLap;
    private double _pendingMs;
    private long _pendingSyncs;
    private double _totalMs;
    private long _totalSyncs;

    private StageTimer(IBackend backend, string model)
    {
        _backend = backend;
        _model = model;
        _syncsAtLap = backend.GetD2hSyncCount();
    }

    /// <summary>A timer whose report is tagged <paramref name="model"/> when <c>diagnostics.profile</c> is on, else null.</summary>
    public static StageTimer? Start(IBackend backend, string model) =>
        EngineKnobs.Profile.Value ? new StageTimer(backend, model) : null;

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
    /// decode step or of one layer); the enclosing <see cref="Mark"/> still counts it.</summary>
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

    /// <summary>Keeps the smallest <paramref name="value"/> seen under <paramref name="name"/> for the report (a decode's
    /// narrowest top-1/top-2 logit gap, say).</summary>
    public void NoteMinimum(string name, float value)
    {
        for (int i = 0; i < _minimums.Count; i++)
        {
            if (_minimums[i].Name == name)
            {
                if (value < _minimums[i].Value)
                {
                    _minimums[i] = (name, value);
                }
                return;
            }
        }
        _minimums.Add((name, value));
    }

    /// <summary>Logs every marked stage, the per-bucket totals and means, the noted minimums and the total under
    /// <paramref name="scope"/>.</summary>
    public void Report(string scope)
    {
        StringBuilder line = new();
        line.Append('[').Append(_model).Append("] ").Append(scope).Append(": ").Append(_stages).Append(" total=")
            .Append(Format(_totalMs)).Append("ms/").Append(_totalSyncs).Append(" (stage=ms/D2H syncs)");
        foreach ((string name, double ms, long syncs, int count) in _buckets)
        {
            line.Append(" | ").Append(name).Append(": n=").Append(count).Append(' ').Append(Format(ms)).Append("ms (")
                .Append(Format(ms / count)).Append(" each) ").Append(syncs).Append(" syncs");
        }
        foreach ((string name, float value) in _minimums)
        {
            line.Append(" | min ").Append(name).Append('=').Append(value.ToString("G6", CultureInfo.InvariantCulture));
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
