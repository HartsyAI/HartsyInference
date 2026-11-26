using System.Globalization;

namespace HartsyInference.Cuda;

/// <summary>Per-run free-memory log for the <c>numerics.tileFromDevice</c> A/B: one aggregate per process, written to stderr at exit and, when the
/// environment names a file, appended to it.</summary>
/// <remarks>Records only when the knob is on, so an ordinary run logs nothing. Three kinds of record: each tiled call's tile height (the score
/// tile's free and total reading and planned and final rows); each executed attention row (<c>row=</c>, counted in RunValidatedAttentionRows:
/// the count to read as calls); and each budget gate's decision (<c>gate=</c>). Gate decisions include planner probes (PlansBatch to
/// PlanAttention, Cheaper, Explain) and rows that never ran, so a gate's <c>decisions</c> is not a call count. The file route exists because <c>dotnet test</c> at default verbosity does not show a test
/// process's stderr; <c>HARTSY_TILE_FREE_LOG</c> names the file, and each process appends one block.</remarks>
internal static class TileFreeLog
{
    /// <summary>The environment variable that names the file each process appends its summary to.</summary>
    internal const string EnvVar = "HARTSY_TILE_FREE_LOG";

    private sealed class GateTally
    {
        public long Decisions, Over, FreeLimited;
    }

    private static readonly object _gate = new();
    private static bool _registered;
    private static long _calls, _halvedCalls;
    private static ulong _minFree = ulong.MaxValue, _maxFree, _total;
    private static readonly SortedSet<long> _heights = new();
    private static readonly SortedDictionary<string, GateTally> _gates = new(StringComparer.Ordinal);
    private static readonly SortedDictionary<string, long> _rows = new(StringComparer.Ordinal);

    /// <summary>Records one executed attention row by its capability id: a call that ran, not a planner probe. This is the count to read as calls.</summary>
    internal static void RecordRow(string rowId)
    {
        lock (_gate)
        {
            Register();
            _rows[rowId] = _rows.TryGetValue(rowId, out long n) ? n + 1 : 1;
        }
    }

    /// <summary>Records one tiled call: the free and total reading it took, the planned tile height, and the height it ran at after any OOM halving.</summary>
    internal static void Record(ulong freeBytes, ulong totalBytes, long plannedRows, long finalRows)
    {
        lock (_gate)
        {
            Register();
            _calls++;
            if (finalRows < plannedRows) _halvedCalls++;
            _minFree = Math.Min(_minFree, freeBytes);
            _maxFree = Math.Max(_maxFree, freeBytes);
            _total = totalBytes;
            _heights.Add(finalRows);
        }
    }

    /// <summary>Records one budget gate's decision. <paramref name="over"/> means the bytes exceeded the budget (the tiled or fallback side). The
    /// budget came from free VRAM, not half the total, when free was below half the total.</summary>
    internal static void RecordGate(string gate, ulong freeBytes, ulong totalBytes, bool over)
    {
        lock (_gate)
        {
            Register();
            if (!_gates.TryGetValue(gate, out GateTally? tally))
            {
                tally = new GateTally();
                _gates[gate] = tally;
            }
            tally.Decisions++;
            if (over) tally.Over++;
            if (freeBytes < totalBytes / 2) tally.FreeLimited++;
        }
    }

    /// <summary>Clears the aggregate. For tests, so a count is the calls made since the reset.</summary>
    internal static void Reset()
    {
        lock (_gate)
        {
            _calls = 0;
            _halvedCalls = 0;
            _minFree = ulong.MaxValue;
            _maxFree = 0;
            _total = 0;
            _heights.Clear();
            _gates.Clear();
            _rows.Clear();
        }
    }

    /// <summary>The aggregate, one line per kind: the tile line first, then one line per gate in name order.</summary>
    internal static string Summary()
    {
        lock (_gate)
        {
            if (_calls == 0 && _gates.Count == 0 && _rows.Count == 0) return "tileFromDevice: no tiled calls, gate decisions or executed rows this process.";
            List<string> lines = [];
            foreach (KeyValuePair<string, long> row in _rows)
            {
                lines.Add($"tileFromDevice row={row.Key} executed={row.Value}");
            }
            if (_calls > 0)
            {
                string heights = string.Join(",", _heights.Take(8)) + (_heights.Count > 8 ? ",..." : "");
                lines.Add($"tileFromDevice: calls={_calls} halved={_halvedCalls} totalMiB={_total >> 20} " +
                          $"freeMiB min={_minFree >> 20} max={_maxFree >> 20} tileRows=[{heights}]");
            }
            foreach (KeyValuePair<string, GateTally> gate in _gates)
            {
                lines.Add($"tileFromDevice gate={gate.Key} decisions={gate.Value.Decisions} over={gate.Value.Over} " +
                          $"freeLimited={gate.Value.FreeLimited}");
            }
            return string.Join('\n', lines);
        }
    }

    /// <summary>Appends the summary to <paramref name="path"/> as one block, with the process id and a UTC stamp. A failure to write is reported on
    /// stderr, never thrown: the log must not fail a run.</summary>
    internal static void AppendTo(string path)
    {
        try
        {
            string stamp = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
            File.AppendAllText(path, $"# pid={Environment.ProcessId} utc={stamp}{Environment.NewLine}{Summary()}{Environment.NewLine}");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            Console.Error.WriteLine($"{EnvVar}: could not append to '{path}': {ex.Message}");
        }
    }

    /// <summary>Registers the process-exit writer once. Caller holds <c>_gate</c>.</summary>
    private static void Register()
    {
        if (_registered) return;
        _registered = true;
        AppDomain.CurrentDomain.ProcessExit += (_, _) => WriteAtExit();
    }

    private static void WriteAtExit()
    {
        string summary = Summary();
        Console.Error.WriteLine(summary);
        string? path = Environment.GetEnvironmentVariable(EnvVar);
        if (!string.IsNullOrWhiteSpace(path)) AppendTo(path);
    }
}
