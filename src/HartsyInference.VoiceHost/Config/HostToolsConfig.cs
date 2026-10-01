using HartsyInference.VoiceHost.Tools;

namespace HartsyInference.VoiceHost.Config;

/// <summary>The tools offered to the model on every call (<c>tools</c> section).</summary>
public sealed record HostToolsConfig
{
    /// <summary>Tool names from <see cref="VoiceHostTools.Names"/>; all of them by default. The gateway's dial plan still
    /// governs <c>transfer</c>.</summary>
    public string[] Enabled { get; set; } = [.. VoiceHostTools.Names];

    /// <summary>How long a telephony tool waits for the gateway's answer before telling the model it failed.</summary>
    public int TimeoutMs { get; set; } = 10_000;
}
