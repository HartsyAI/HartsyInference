using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Core.Backends;

/// <summary>Inclusive device time per named model stage, accumulated per process while <see cref="EngineKnobs.Profile"/> is on.
/// A scope syncs the backend when it opens and when it closes, so its time is the stage's GPU time with nested scopes
/// included. Off by default; when off a scope reads the knob and does nothing else.</summary>
/// <remarks>The per-op table of <c>diagnostics.profile</c> attributes time to backend calls; this one attributes it to the
/// model's own stages, the granularity a layer-by-layer comparison against a reference implementation needs. The syncs
/// serialize host and device, so totals under the profiler run slower than an unprofiled pass.</remarks>
public static class StageProfiler
{
    private static readonly ConcurrentDictionary<string, long[]> _stats = new(StringComparer.Ordinal);

    /// <summary>Whether scopes record. Reads <see cref="EngineKnobs.Profile"/>.</summary>
    public static bool Enabled => EngineKnobs.Profile.Value;

    /// <summary>Opens a scope labelled <c>prefix.stage</c>; the label is built only when the profiler is on.</summary>
    public static Scope Begin(IBackend backend, string prefix, string stage)
    {
        if (!Enabled)
            return default;
        ArgumentNullException.ThrowIfNull(backend);
        backend.Sync();
        return new Scope(backend, string.Concat(prefix, ".", stage), Stopwatch.GetTimestamp());
    }

    /// <summary>Drops everything recorded so far.</summary>
    public static void Reset() => _stats.Clear();

    /// <summary>Recorded stages as (label, calls, total milliseconds), in label order.</summary>
    public static IReadOnlyList<(string Label, long Calls, double TotalMs)> Snapshot()
    {
        double toMs = 1000.0 / Stopwatch.Frequency;
        return _stats.OrderBy(e => e.Key, StringComparer.Ordinal)
            .Select(e => (e.Key, Volatile.Read(ref e.Value[0]), Volatile.Read(ref e.Value[1]) * toMs))
            .ToList();
    }

    /// <summary>The snapshot as a tab-separated table with a header line.</summary>
    public static string Report()
    {
        StringBuilder text = new("stage\tcalls\ttotal_ms\n");
        foreach ((string label, long calls, double totalMs) in Snapshot())
            text.Append(CultureInfo.InvariantCulture, $"{label}\t{calls}\t{totalMs:F2}\n");
        return text.ToString();
    }

    /// <summary>One open stage. Disposing it syncs the backend and adds the elapsed time to its label.</summary>
    public readonly ref struct Scope
    {
        private readonly IBackend? _backend;
        private readonly string? _label;
        private readonly long _start;

        internal Scope(IBackend backend, string label, long start)
        {
            _backend = backend;
            _label = label;
            _start = start;
        }

        /// <summary>Closes the scope.</summary>
        public void Dispose()
        {
            if (_label is null)
                return;
            _backend!.Sync();
            long elapsed = Stopwatch.GetTimestamp() - _start;
            long[] slot = _stats.GetOrAdd(_label, static _ => new long[2]);
            Interlocked.Increment(ref slot[0]);
            Interlocked.Add(ref slot[1], elapsed);
        }
    }
}
