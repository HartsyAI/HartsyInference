namespace HartsyInference.Cuda;

/// <summary>Per-run free-memory log for the <c>numerics.tileFromDevice</c> A/B: one aggregate per process, written to stderr at exit.</summary>
/// <remarks>Records only when the knob is on, so an ordinary run logs nothing. Free memory is logged so a run can be read against the gates it
/// might have tripped: a call's tile height is the device's, but the gate that picks its path still reads free VRAM under the clamp.</remarks>
internal static class TileFreeLog
{
    private static readonly object _gate = new();
    private static bool _registered;
    private static long _calls, _halvedCalls;
    private static ulong _minFree = ulong.MaxValue, _maxFree, _total;
    private static readonly SortedSet<long> _heights = new();

    /// <summary>Records one tiled call: the free and total reading it took, the planned tile height, and the height it ran at after any OOM halving.</summary>
    internal static void Record(ulong freeBytes, ulong totalBytes, long plannedRows, long finalRows)
    {
        lock (_gate)
        {
            if (!_registered)
            {
                _registered = true;
                AppDomain.CurrentDomain.ProcessExit += (_, _) => WriteSummary();
            }
            _calls++;
            if (finalRows < plannedRows) _halvedCalls++;
            _minFree = Math.Min(_minFree, freeBytes);
            _maxFree = Math.Max(_maxFree, freeBytes);
            _total = totalBytes;
            _heights.Add(finalRows);
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
        }
    }

    /// <summary>The one-line aggregate: call count, OOM-halved calls, the free-VRAM range in MiB, and the distinct tile heights run.</summary>
    internal static string Summary()
    {
        lock (_gate)
        {
            if (_calls == 0) return "tileFromDevice: no tiled calls this process.";
            string heights = string.Join(",", _heights.Take(8)) + (_heights.Count > 8 ? ",..." : "");
            return $"tileFromDevice: calls={_calls} halved={_halvedCalls} totalMiB={_total >> 20} " +
                   $"freeMiB min={_minFree >> 20} max={_maxFree >> 20} tileRows=[{heights}]";
        }
    }

    private static void WriteSummary() => Console.Error.WriteLine(Summary());
}
