using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Runtime;

/// <summary>Watches a stage that reports each finished step through <see cref="Step"/>. When no step finishes for much longer than the steps so far took (<paramref name="stallFactor"/> times their median, but never under <paramref name="minimumStall"/>), it logs one warning, calls <paramref name="onStall"/>, and makes the next <see cref="Step"/> throw <see cref="StageStalledException"/>. A step that is still running cannot be interrupted, so a stage that never returns is reported but not ended; the host decides whether to stop the process.</summary>
public sealed class StageWatchdog : IDisposable
{
    private readonly string _stage;
    private readonly Func<(long FreeBytes, long TotalBytes)>? _memory;
    private readonly double _stallFactor;
    private readonly double _minimumStallSeconds;
    private readonly Action<StageStalledException>? _onStall;
    private readonly Func<double> _nowSeconds;
    private readonly Timer? _timer;
    private readonly object _gate = new();
    private readonly List<double> _stepSeconds = [];
    private double _lastProgress;
    private StageStalledException? _stalled;
    private bool _disposed;

    /// <param name="stage">The name the warning and the exception carry.</param>
    /// <param name="memory">Reads the device's free and total bytes when a stall is recorded; null leaves them out of the message.</param>
    /// <param name="stallFactor">How many times the median step counts as a stall.</param>
    /// <param name="minimumStall">The shortest silence ever called a stall, which also covers the first step (the weights load inside it).</param>
    /// <param name="onStall">Called once, on the timer thread, when a stall is recorded.</param>
    /// <param name="pollEvery">How often the timer checks; <see cref="TimeSpan.Zero"/> starts no timer and the caller drives <see cref="Poll"/>.</param>
    /// <param name="nowSeconds">The clock in seconds; defaults to the monotonic clock.</param>
    public StageWatchdog(
        string stage, Func<(long FreeBytes, long TotalBytes)>? memory = null, double stallFactor = 10, TimeSpan? minimumStall = null,
        Action<StageStalledException>? onStall = null, TimeSpan? pollEvery = null, Func<double>? nowSeconds = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stage);
        ArgumentOutOfRangeException.ThrowIfLessThan(stallFactor, 1.0);
        _stage = stage;
        _memory = memory;
        _stallFactor = stallFactor;
        _minimumStallSeconds = (minimumStall ?? TimeSpan.FromMinutes(2)).TotalSeconds;
        _onStall = onStall;
        _nowSeconds = nowSeconds ?? (() => MonotonicClock.NowNs() / 1e9);
        _lastProgress = _nowSeconds();
        TimeSpan period = pollEvery ?? TimeSpan.FromSeconds(1);
        if (period > TimeSpan.Zero)
            _timer = new Timer(_ => Poll(), null, period, period);
    }

    /// <summary>Records a finished step.</summary>
    /// <exception cref="StageStalledException">The stage had stalled before this step finished.</exception>
    public void Step()
    {
        lock (_gate)
        {
            double now = _nowSeconds();
            _stepSeconds.Add(now - _lastProgress);
            _lastProgress = now;
            if (_stalled is not null)
                throw _stalled;
        }
    }

    /// <summary>Checks the silence against the limit and records the stall the first time it is exceeded; the timer calls it.</summary>
    public void Poll()
    {
        StageStalledException? raised = null;
        lock (_gate)
        {
            if (_disposed || _stalled is not null)
                return;
            double silence = _nowSeconds() - _lastProgress;
            double expected = Median();
            if (silence <= Math.Max(_minimumStallSeconds, _stallFactor * expected))
                return;
            (long free, long total) = _memory?.Invoke() ?? (0, 0);
            _stalled = raised = new StageStalledException(_stage, silence, expected, free, total);
        }

        Logs.Warning($"[Stall] {raised.Message}");
        _onStall?.Invoke(raised);
    }

    /// <summary>Stops the timer.</summary>
    public void Dispose()
    {
        lock (_gate)
            _disposed = true;
        _timer?.Dispose();
    }

    private double Median()
    {
        if (_stepSeconds.Count == 0)
            return 0;
        double[] sorted = [.. _stepSeconds];
        Array.Sort(sorted);
        return sorted[sorted.Length / 2];
    }
}
