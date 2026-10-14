using System.Globalization;

namespace HartsyInference.Tests.Common;

/// <summary>Which Vulkan device the test harness pins to. <c>HARTSY_TEST_VULKAN_DEVICE</c> names an ordinal; when it
/// is set, every harness path opens exactly that device and none of them probes device 0 or falls back to the
/// backend's default picker. On the dev box both of those are the RTX 4090, which runs a live SwarmUI, so a silent
/// fallback there is a real hazard. Unset keeps the default behaviour.</summary>
public static class VulkanTestDeviceSelection
{
    /// <summary>The environment variable that names the ordinal.</summary>
    public const string EnvVar = "HARTSY_TEST_VULKAN_DEVICE";

    /// <summary>The ordinal the environment pins, or null when the variable is unset or blank.</summary>
    public static int? FromEnvironment() => Parse(Environment.GetEnvironmentVariable(EnvVar));

    /// <summary>Parses a raw value. Null, empty or whitespace means unset. Anything else must be a plain non-negative
    /// integer; a value that names no device throws instead of falling back, so a typo cannot land on the 4090.</summary>
    public static int? Parse(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }
        if (int.TryParse(raw.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out int ordinal))
        {
            return ordinal;
        }
        throw new FormatException($"{EnvVar}='{raw}' is not a Vulkan device ordinal (a non-negative integer).");
    }
}
