using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Runtime;

/// <summary>Watches a stage that reports each finished step through <see cref="Step"/>. When no step finishes for much longer than the steps so far took (<c>stallFactor</c> times their median, but never under <c>minimumStall</c>), it logs one warning, calls the stall callback, and makes the next <see cref="Step"/> throw <see cref="StageStalledException"/>. A step that is still running cannot be interrupted, so a stage that never returns is reported but not ended; the host decides whether to stop the process. The device memory is read on the thread that calls <see cref="Step"/>, never from the timer, because a backend query can wait for the lock a long operation holds. A device that reports no memory (the CPU backend) is not watched, and a slow first step (lazy uploads and kernel setup after the weights are placed) only warns and is left out of the median.</summary>
public sealed class StageWatchdog : IDisposable
{
    private readonly string _stage;
    private readonly Func<(long FreeBytes, long TotalBytes)>? _memory;
    private readonly double _stallFactor;
    private readonly double _minimumStallSeconds;
    private readonly string? _remedy;
    private readonly Action<StageStalledException>? _onStall;
    private readonly Func<double> _nowSeconds;
    private readonly Timer? _timer;
    private readonly object _gate = new();
    private readonly List<double> _stepSeconds = [];
    private readonly bool _watched;
    private double _lastProgress;
    private (long Free, long Total) _snapshot;
    private StageStalledException? _stalled;
    private int _polling;
    private bool _disposed;

    /// <param name="stage">The name the warning and the exception carry.</param>
    /// <param name="memory">Reads the device's free and total bytes; sampled when the watchdog is made and after each step, on the calling thread. Null, or a total of 0, leaves the stage unwatched.</param>
    /// <param name="stallFactor">How many times the median step counts as a stall.</param>
    /// <param name="minimumStall">The shortest silence ever called a stall, which also covers the first step.</param>
    /// <param name="remedy">What the caller can try, in the stage's own terms; it goes into the warning and the exception.</param>
    /// <param name="onStall">Called once per stall, on the timer thread.</param>
    /// <param name="pollEvery">How often the timer checks; <see cref="TimeSpan.Zero"/> starts no timer and the caller drives <see cref="Poll"/>.</param>
    /// <param name="nowSeconds">The clock in seconds; defaults to the monotonic clock.</param>
    public StageWatchdog(
        string stage, Func<(long FreeBytes, long TotalBytes)>? memory = null, double stallFactor = 10, TimeSpan? minimumStall = null,
        string? remedy = null, Action<StageStalledException>? onStall = null, TimeSpan? pollEvery = null, Func<double>? nowSeconds = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stage);
        ArgumentOutOfRangeException.ThrowIfLessThan(stallFactor, 1.0);
        _stage = stage;
        _memory = memory;
        _stallFactor = stallFactor;
        _minimumStallSeconds = (minimumStall ?? TimeSpan.FromMinutes(2)).TotalSeconds;
        _remedy = remedy;
        _onStall = onStall;
        _nowSeconds = nowSeconds ?? (() => MonotonicClock.NowNs() / 1e9);
        _snapshot = memory?.Invoke() ?? (0, 0);
        _watched = _snapshot.Total > 0;
        _lastProgress = _nowSeconds();
        TimeSpan period = pollEvery ?? TimeSpan.FromSeconds(1);
        if (_watched && period > TimeSpan.Zero)
            _timer = new Timer(_ => Poll(), null, period, period);
    }

    /// <summary>Records a finished step.</summary>
    /// <exception cref="StageStalledException">The stage had stalled before this step finished, and this is not its first step.</exception>
    public void Step()
    {
        // On the calling thread on purpose: on a device backend the query takes the same lock a running operation holds, so the timer must not make it.
        (long, long)? sampled = _watched ? _memory!.Invoke() : null;
        lock (_gate)
        {
            double now = _nowSeconds();
            _stepSeconds.Add(now - _lastProgress);
            _lastProgress = now;
            if (sampled is { } memory)
                _snapshot = memory;
            StageStalledException? stalled = _stalled;
            if (stalled is null)
                return;
            // The first step sets up lazily (uploads, kernels); a long one is reported but does not end the run.
            if (_stepSeconds.Count == 1)
            {
                _stalled = null;
                return;
            }

            throw new StageStalledException(_stage, stalled.StalledSeconds, stalled.ExpectedStepSeconds, stalled.FreeBytes, stalled.TotalBytes, _remedy, stalled);
        }
    }

    /// <summary>Checks the silence against the limit and records the stall the first time it is exceeded; the timer calls it. It never calls the memory delegate.</summary>
    public void Poll()
    {
        if (Interlocked.Exchange(ref _polling, 1) == 1)
            return;
        try
        {
            StageStalledException? raised = null;
            lock (_gate)
            {
                if (_disposed || !_watched || _stalled is not null)
                    return;
                double silence = _nowSeconds() - _lastProgress;
                double expected = Median();
                if (silence <= Math.Max(_minimumStallSeconds, _stallFactor * expected))
                    return;
                _stalled = raised = new StageStalledException(_stage, silence, expected, _snapshot.Free, _snapshot.Total, _remedy);
            }

            Logs.Warning($"[Stall] {raised.Message}");
            if (_onStall is not null)
            {
                // A callback that throws on the timer thread would end the process, and the watchdog only reports.
                try
                {
                    _onStall(raised);
                }
                catch (Exception ex)
                {
                    Logs.Error($"[Stall] The stall callback for '{_stage}' threw {ex.GetType().Name} and was ignored: {ex.Message}");
                }
            }
        }
        finally
        {
            Volatile.Write(ref _polling, 0);
        }
    }

    /// <summary>Stops the timer.</summary>
    public void Dispose()
    {
        lock (_gate)
            _disposed = true;
        _timer?.Dispose();
    }

    // The first step loads and sets up, so once a later step exists it does not count towards what a step normally takes.
    private double Median()
    {
        if (_stepSeconds.Count == 0)
            return 0;
        double[] sorted = _stepSeconds.Count == 1 ? [_stepSeconds[0]] : [.. _stepSeconds.Skip(1)];
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
    }
}
