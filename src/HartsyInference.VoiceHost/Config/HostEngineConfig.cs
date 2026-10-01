namespace HartsyInference.VoiceHost.Config;

/// <summary>Engine process settings (<c>engine</c> section).</summary>
public sealed record HostEngineConfig
{
    /// <summary>Caps the engine's CPU kernel threads (<c>numerics.cpuThreads</c>) for the host's lifetime; 0 leaves the
    /// engine setting alone. Match it to the CPUs the unit allows.</summary>
    public int CpuThreadCap { get; set; }

    /// <summary>Absolute path of an engine settings file (<c>paths.modelsRoot</c> and the rest) read instead of the
    /// service user's <c>~/.config/hartsyinference/settings.json</c>; empty uses that one.</summary>
    public string SettingsFile { get; set; } = "";
}
