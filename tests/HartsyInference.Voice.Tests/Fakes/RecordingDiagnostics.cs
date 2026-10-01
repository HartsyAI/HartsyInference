using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Engine.Diagnostics;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>An <see cref="IInferenceDiagnostics"/> that records every event with a wall-clock timestamp, for measuring
/// what a real generation does token by token (TTFT, decode cadence, prompt token count) rather than inferring it
/// from the chunks a stream filter forwards.</summary>
internal sealed class RecordingDiagnostics : IInferenceDiagnostics
{
    private readonly List<Entry> _events = [];
    private readonly object _lock = new();
    private readonly Stopwatch _clock = Stopwatch.StartNew();

    /// <summary>One recorded event: the request id the engine correlates a generation's events with, the stage, a
    /// count (prompt tokens for <see cref="InferenceDiagnosticKind.PrefillCompleted"/>, the running token index for
    /// <see cref="InferenceDiagnosticKind.TokenGenerated"/>), and the wall-clock moment it fired.</summary>
    public readonly record struct Entry(long RequestId, InferenceDiagnosticKind Kind, int Count, double ElapsedMs);

    /// <summary>Every event recorded so far, in arrival order.</summary>
    public IReadOnlyList<Entry> Events
    {
        get
        {
            lock (_lock)
            {
                return [.. _events];
            }
        }
    }

    /// <inheritdoc/>
    public void OnEvent(in InferenceDiagnosticEvent diagnostic)
    {
        Entry entry = new(diagnostic.RequestId, diagnostic.Kind, diagnostic.Count, _clock.Elapsed.TotalMilliseconds);
        lock (_lock)
        {
            _events.Add(entry);
        }
    }

    /// <inheritdoc/>
    public void OnBackendReady(long requestId, IBackend backend)
    {
        // Not needed for timing; the backend itself is read separately (CudaPlanStats) after warm-up and after
        // each turn, which is cheaper than snapshotting plan counters on every request boundary.
    }

    /// <summary>Events grouped by request id, each group in the order <see cref="OnEvent"/> saw it, groups in the
    /// order their first event arrived — i.e. one group per <c>GenerateAsync</c>/<c>StreamAsync</c> call, in call
    /// order (warm-up first, then turn 1's call(s), then turn 2's, …).</summary>
    public IReadOnlyList<IReadOnlyList<Entry>> ByRequest()
    {
        List<Entry> snapshot = [.. Events];
        List<long> order = [];
        Dictionary<long, List<Entry>> groups = [];
        foreach (Entry entry in snapshot)
        {
            if (!groups.TryGetValue(entry.RequestId, out List<Entry>? list))
            {
                list = [];
                groups[entry.RequestId] = list;
                order.Add(entry.RequestId);
            }
            list.Add(entry);
        }
        return [.. order.Select(id => (IReadOnlyList<Entry>)groups[id])];
    }

    /// <summary>A plain-language summary of one request's events: TTFT (first <see cref="InferenceDiagnosticKind.TokenGenerated"/>
    /// minus <see cref="InferenceDiagnosticKind.RequestStarted"/>), prompt tokens (<see cref="InferenceDiagnosticKind.PrefillCompleted"/>'s
    /// count), tokens generated and decode tok/s over them (excluding the first, which prefill already paid for).</summary>
    public static string Describe(IReadOnlyList<Entry> request)
    {
        double? started = request.Where(e => e.Kind == InferenceDiagnosticKind.RequestStarted).Select(e => (double?)e.ElapsedMs).FirstOrDefault();
        Entry? prefill = request.Where(e => e.Kind == InferenceDiagnosticKind.PrefillCompleted).Select(e => (Entry?)e).FirstOrDefault();
        List<Entry> tokens = [.. request.Where(e => e.Kind == InferenceDiagnosticKind.TokenGenerated)];
        if (started is null || tokens.Count == 0)
        {
            return "no token events recorded";
        }
        double ttftMs = tokens[0].ElapsedMs - started.Value;
        string decode = tokens.Count >= 2
            ? $", decode {(tokens.Count - 1) / ((tokens[^1].ElapsedMs - tokens[0].ElapsedMs) / 1000.0):F1} tok/s over {tokens.Count} tokens"
            : $", {tokens.Count} token";
        string prompt = prefill is { } p ? $", prompt {p.Count} tok" : "";
        return $"ttft {ttftMs:F1} ms{decode}{prompt}";
    }
}
