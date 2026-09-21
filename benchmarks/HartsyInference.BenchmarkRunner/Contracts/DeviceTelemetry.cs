using System.Text.Json.Serialization;

namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Device telemetry aggregated over one measured request. Contributor disclosure, not recomputable
/// evidence: unlike the token traces, nothing here can be re-derived from the outputs.</summary>
public sealed record DeviceTelemetry
{
    public const string Unavailable = "unavailable";
    public const string DeviceWide = "nvidia-smi-device-wide";
    public const string PerProcess = "nvidia-smi-process";
    public static readonly string[] Sources = [Unavailable, DeviceWide, PerProcess];

    public required string Source { get; init; }
    public int SampleCount { get; init; }
    /// <summary>Largest gap between consecutive samples. A sampler that died mid-request reads as a gap here
    /// rather than as a clean aggregate over the part it did see.</summary>
    public double MaxSampleIntervalMs { get; init; }
    /// <summary>Device-wide unless <see cref="Source"/> is <see cref="PerProcess"/>; honest as a per-run figure
    /// only because the campaign refused to start on a shared device.</summary>
    public long? PeakUsedDeviceBytes { get; init; }
    public double? PeakGpuUtilizationPercent { get; init; }
    public double? MeanGpuUtilizationPercent { get; init; }
    public double? MeanMemoryUtilizationPercent { get; init; }
    public double? PeakPowerWatts { get; init; }
    public double? MeanPowerWatts { get; init; }
    public double? MaxTemperatureCelsius { get; init; }
    public int? MinSmClockMhz { get; init; }
    public double? MeanSmClockMhz { get; init; }
    public int HwSlowdownSamples { get; init; }
    public int HwThermalSamples { get; init; }
    public int HwPowerBrakeSamples { get; init; }
    public int SwThermalSamples { get; init; }
    /// <summary>Counted and shown, never disqualifying: a stock card under sustained GEMM load sits at its
    /// software power cap for most of a run.</summary>
    public int SwPowerCapSamples { get; init; }

    /// <summary>Samples carrying a reason that makes a run incomparable. <c>gpu_idle</c> is excluded because an
    /// idle card reports it continuously, and <c>sw_power_cap</c> because normal load reports it.</summary>
    [JsonIgnore]
    public int ThrottledSamples => HwSlowdownSamples + HwThermalSamples + HwPowerBrakeSamples + SwThermalSamples;

    [JsonIgnore]
    public double ThrottledFraction => SampleCount > 0 ? (double)ThrottledSamples / SampleCount : 0;
}
