using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Core.Memory.Vram;

/// <summary>Appends labelled device-memory readings to the file <see cref="EngineKnobs.VramTrace"/> names, one tab-separated
/// line per reading in MiB: driver free, used and total, the allocator pool's reserved and used bytes, resident weights and
/// activations, then milliseconds since the first reading. Off by default; when off a call reads the knob and returns.</summary>
/// <remarks>Each reading follows a device sync, so stream-ordered frees have landed and the pool figures are settled. The
/// driver figures are device-wide, other processes included; reserved minus used is memory the pool holds but nothing uses.</remarks>
public static class DeviceMemoryTrace
{
    private const string Header =
        "label\tfree_mib\tused_mib\ttotal_mib\tpool_reserved_mib\tpool_used_mib\tweights_mib\tactivations_mib\tms\n";
    private const double Mib = 1024.0 * 1024.0;

    private static readonly object _gate = new();
    private static readonly Stopwatch _clock = new();

    /// <summary>Whether <see cref="EngineKnobs.VramTrace"/> names a file. Callers that compose a label check it first.</summary>
    public static bool Enabled => EngineKnobs.VramTrace.Value is not null;

    /// <summary>The file a model with its own older trace knob writes to: <see cref="EngineKnobs.VramTrace"/> when set, else
    /// <paramref name="legacy"/>'s value.</summary>
    public static string? FilePath(Knob<string?> legacy)
    {
        ArgumentNullException.ThrowIfNull(legacy);
        return EngineKnobs.VramTrace.Value ?? legacy.Value;
    }

    /// <summary>Records one reading under <paramref name="label"/> when the trace is on.</summary>
    public static void Record(IBackend backend, string label)
    {
        if (EngineKnobs.VramTrace.Value is string path)
            RecordTo(path, backend, label);
    }

    /// <summary>Records under <c>label index</c>, formatting the label only when the trace is on.</summary>
    public static void Record(IBackend backend, string label, int index)
    {
        if (EngineKnobs.VramTrace.Value is string path)
            RecordTo(path, backend, string.Create(CultureInfo.InvariantCulture, $"{label} {index}"));
    }

    /// <summary>Records under <c>label: detail</c>, joining the two only when the trace is on.</summary>
    public static void Record(IBackend backend, string label, string detail)
    {
        if (EngineKnobs.VramTrace.Value is string path)
            RecordTo(path, backend, string.Concat(label, ": ", detail));
    }

    /// <summary>Syncs <paramref name="backend"/> and appends its reading under <paramref name="label"/> to <paramref name="path"/>,
    /// writing the column header first when the file is new or empty.</summary>
    public static void RecordTo(string path, IBackend backend, string label)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(label);
        backend.Sync();
        DeviceMemoryUsage usage = backend.GetDeviceMemoryUsage();
        lock (_gate)
        {
            if (!_clock.IsRunning)
                _clock.Start();
            string line = string.Join('\t', label, Format(usage.FreeBytes), Format(Used(usage)), Format(usage.TotalBytes),
                Format(usage.PoolReservedBytes), Format(usage.PoolUsedBytes), Format(usage.WeightBytes), Format(usage.ActivationBytes),
                _clock.ElapsedMilliseconds.ToString(CultureInfo.InvariantCulture)) + "\n";
            FileInfo file = new(path);
            File.AppendAllText(path, file.Exists && file.Length > 0 ? line : Header + line);
        }
    }

    private static long Used(DeviceMemoryUsage usage) =>
        usage.FreeBytes < 0 || usage.TotalBytes <= 0 ? -1 : usage.TotalBytes - usage.FreeBytes;

    private static string Format(long bytes) =>
        bytes < 0 ? "-" : (bytes / Mib).ToString("F1", CultureInfo.InvariantCulture);
}
