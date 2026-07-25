namespace HartsyInference.Core.Runtime;

/// <summary>Tri-state environment switches backing the engine's standard performance profile (<c>docs/PERFORMANCE.md</c>).</summary>
/// <remarks>Proven profile features resolve their default from code — ON for every install, so published
/// benchmark times reproduce without any configuration — while an explicit environment value always wins:
/// <c>"1"</c>/<c>"true"</c> forces a feature on, <c>"0"</c>/<c>"false"</c> is the supported kill-switch for
/// debugging and constrained hardware. Experimental switches keep the legacy strict opt-in form (<c>== "1"</c>)
/// at their call sites and are not routed through here.</remarks>
public static class EnvSwitch
{
    /// <summary>Resolves <paramref name="name"/> from the environment, defaulting to <paramref name="defaultOn"/> when unset/unrecognized.</summary>
    /// <remarks>Unset or empty → <paramref name="defaultOn"/>; <c>"1"</c>/<c>"true"</c> (any case) → <c>true</c>;
    /// <c>"0"</c>/<c>"false"</c> (any case) → <c>false</c>; any other value → <paramref name="defaultOn"/>.</remarks>
    public static bool IsEnabled(string name, bool defaultOn)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(value))
        {
            return defaultOn;
        }
        if (value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        if (value == "0" || value.Equals("false", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        return defaultOn;
    }
}
