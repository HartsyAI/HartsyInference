using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>Stage-level wall-clock attribution for one Kokoro synthesis, active only under <c>diagnostics.profile</c>.
/// Each <see cref="Mark"/> drains the backend stream first, so a stage's time is its true host + device cost rather
/// than its async launch cost — that serialization is why this stays off by default. Null when profiling is off,
/// so call sites cost one null check.</summary>
internal sealed class KokoroStageTimer
{
    private readonly IBackend _backend;
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly StringBuilder _report = new();
    private double _totalMs;

    private KokoroStageTimer(IBackend backend)
    {
        _backend = backend;
    }

    /// <summary>A timer when <c>diagnostics.profile</c> is on, else null.</summary>
    public static KokoroStageTimer? Start(IBackend backend) => EngineKnobs.Profile.Value ? new KokoroStageTimer(backend) : null;

    /// <summary>Closes the stage named <paramref name="stage"/>: syncs the device, records the elapsed time, restarts the clock.</summary>
    public void Mark(string stage)
    {
        _backend.Sync();
        double ms = _stopwatch.Elapsed.TotalMilliseconds;
        _totalMs += ms;
        if (_report.Length > 0)
        {
            _report.Append(' ');
        }
        _report.Append(stage).Append('=').Append(ms.ToString("F1", CultureInfo.InvariantCulture));
        _stopwatch.Restart();
    }

    /// <summary>Logs every marked stage and the total under <paramref name="scope"/>.</summary>
    public void Report(string scope) =>
        Logs.Info($"[Kokoro] {scope}: {_report} total={_totalMs.ToString("F1", CultureInfo.InvariantCulture)} ms");
}
