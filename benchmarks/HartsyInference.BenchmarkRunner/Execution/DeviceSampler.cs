using System.Diagnostics;
using System.Globalization;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>One long-lived <c>nvidia-smi -lms</c> per session, aggregated per measured request. Samples are
/// stamped by the reader with <see cref="Stopwatch"/> ticks rather than read from nvidia-smi's own wall-clock
/// column, because the measurement window is in ticks and the two clocks are not comparable.</summary>
public sealed class DeviceSampler : IDisposable
{
    private static readonly string[] Queried =
    [
        "utilization.gpu", "utilization.memory", "memory.used", "temperature.gpu", "power.draw", "clocks.sm",
        "clocks_throttle_reasons.hw_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_power_brake_slowdown", "clocks_throttle_reasons.sw_power_cap",
        "clocks_throttle_reasons.sw_thermal_slowdown"
    ];

    private const int MaximumSamples = 65536;

    private readonly List<Sample> _samples = [];
    private readonly object _gate = new();
    private readonly Process? _process;
    private readonly Thread? _reader;
    private readonly Task<string>? _errors;
    private readonly string _source;
    private string? _note;

    private DeviceSampler(Process? process, string source, string? note)
    {
        _process = process;
        _source = source;
        _note = note;
        if (process is null)
            return;
        _errors = process.StandardError.ReadToEndAsync();
        _reader = new Thread(Read) { IsBackground = true, Name = "hartsy-bench-telemetry" };
        _reader.Start();
    }

    /// <summary>Why telemetry is missing or partial, when it is.</summary>
    public string? Note => _note;

    /// <summary>Starts sampling, or returns an inert sampler when the device could not be attested. A missing
    /// sampler is recorded as unavailable telemetry, never as a failed session.</summary>
    public static DeviceSampler Start(string? uuid, int cadenceMs)
    {
        if (uuid is null)
            return new DeviceSampler(null, DeviceTelemetry.Unavailable, "Device was not attested by nvidia-smi.");
        Process? process = NvidiaSmi.Start(["-i", uuid, "--query-gpu=" + string.Join(',', Queried), "--format=csv,noheader,nounits",
            "-lms", cadenceMs.ToString(CultureInfo.InvariantCulture)]);
        return process is null
            ? new DeviceSampler(null, DeviceTelemetry.Unavailable, "nvidia-smi could not be started.")
            : new DeviceSampler(process, DeviceTelemetry.DeviceWide, null);
    }

    private void Read()
    {
        try
        {
            while (_process!.StandardOutput.ReadLine() is { } line)
            {
                long ticks = Stopwatch.GetTimestamp();
                string[] row = NvidiaSmi.Fields(line);
                if (row.Length != Queried.Length)
                    continue;
                Sample sample = new(ticks, Number(row[0]), Number(row[1]), Number(row[2]) is { } mib ? (long)(mib * 1024 * 1024) : null,
                    Number(row[3]), Number(row[4]), Number(row[5]) is { } clock ? (int)clock : null, Active(row[6]), Active(row[7]),
                    Active(row[8]), Active(row[9]), Active(row[10]));
                lock (_gate)
                {
                    if (_samples.Count < MaximumSamples)
                        _samples.Add(sample);
                }
            }
        }
        catch (Exception error)when (error is not OutOfMemoryException)
        {
            _note ??= "Telemetry stream ended early: " + error.GetType().Name;
        }
    }

    /// <summary>Aggregates the samples inside one measured request. Coverage spans the whole window, so a
    /// sampler that started late or died partway reads as a gap instead of as a clean partial aggregate.</summary>
    public DeviceTelemetry Aggregate(long startTicks, long endTicks)
    {
        Sample[] window;
        lock (_gate)
        {
            // Steps are sequential, so anything before this window can never be needed again.
            _samples.RemoveAll(s => s.Ticks < startTicks);
            window = _samples.Where(s => s.Ticks <= endTicks).ToArray();
        }

        if (_process?.HasExited == true)
            _note ??= "Telemetry sampler exited before the session completed.";
        if (_source == DeviceTelemetry.Unavailable || window.Length == 0)
            return new DeviceTelemetry
            {
                Source = DeviceTelemetry.Unavailable,
                MaxSampleIntervalMs = 0
            };
        double gap = Milliseconds(window[0].Ticks - startTicks);
        for (int i = 1; i < window.Length; i++)
            gap = Math.Max(gap, Milliseconds(window[i].Ticks - window[i - 1].Ticks));
        gap = Math.Max(gap, Milliseconds(endTicks - window[^1].Ticks));
        return new DeviceTelemetry
        {
            Source = _source,
            SampleCount = window.Length,
            MaxSampleIntervalMs = gap,
            PeakUsedDeviceBytes = Extreme(window, s => s.UsedBytes, high: true),
            PeakGpuUtilizationPercent = Peak(window, s => s.GpuUtilization),
            MeanGpuUtilizationPercent = Mean(window, s => s.GpuUtilization),
            MeanMemoryUtilizationPercent = Mean(window, s => s.MemoryUtilization),
            PeakPowerWatts = Peak(window, s => s.Power),
            MeanPowerWatts = Mean(window, s => s.Power),
            MaxTemperatureCelsius = Peak(window, s => s.Temperature),
            MinSmClockMhz = (int?)Extreme(window, s => s.SmClock, high: false),
            MeanSmClockMhz = Mean(window, s => s.SmClock),
            HwSlowdownSamples = window.Count(s => s.HwSlowdown),
            HwThermalSamples = window.Count(s => s.HwThermal),
            HwPowerBrakeSamples = window.Count(s => s.HwPowerBrake),
            SwThermalSamples = window.Count(s => s.SwThermal),
            SwPowerCapSamples = window.Count(s => s.SwPowerCap)
        };
    }

    public void Dispose()
    {
        if (_process is not null)
        {
            NvidiaSmi.Kill(_process);
            _reader?.Join(5000);
            _errors?.Wait(1000);
            _process.Dispose();
        }
    }

    private static double Milliseconds(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
    private static long? Extreme(Sample[] window, Func<Sample, long?> selector, bool high)
    {
        long[] values = window.Select(selector).Where(v => v is not null).Select(v => v!.Value).ToArray();
        return values.Length == 0 ? null : high ? values.Max() : values.Min();
    }

    private static long? Extreme(Sample[] window, Func<Sample, int?> selector, bool high) =>
        Extreme(window, s => (long?)selector(s), high);

    private static double? Peak(Sample[] window, Func<Sample, double?> selector)
    {
        double[] values = window.Select(selector).Where(v => v is not null).Select(v => v!.Value).ToArray();
        return values.Length > 0 ? values.Max() : null;
    }

    private static double? Mean(Sample[] window, Func<Sample, double?> selector)
    {
        double[] values = window.Select(selector).Where(v => v is not null).Select(v => v!.Value).ToArray();
        return values.Length > 0 ? values.Average() : null;
    }

    private static double? Mean(Sample[] window, Func<Sample, int?> selector) => Mean(window, s => (double?)selector(s));
    private static double? Number(string value) => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture,
        out double parsed) ? parsed : null;
    /// <summary>nvidia-smi prints throttle reasons as <c>Active</c> / <c>Not Active</c>, and <c>[N/A]</c> where
    /// the card does not report one.</summary>
    private static bool Active(string value) => value.Equals("Active", StringComparison.OrdinalIgnoreCase);

    private readonly record struct Sample(long Ticks, double? GpuUtilization, double? MemoryUtilization, long? UsedBytes,
        double? Temperature, double? Power, int? SmClock, bool HwSlowdown, bool HwThermal, bool HwPowerBrake, bool SwPowerCap,
        bool SwThermal);
}
