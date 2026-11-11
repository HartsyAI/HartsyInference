namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) when the TRELLIS.2 files are not installed, or when
/// <paramref name="requireEnvironmentVariable"/> names a variable that is not set to 1. The folder is found from
/// <c>HARTSYINFERENCE_MODELS</c>, else by walking up from the test binary to a <c>Models/3D/trellis2/pipeline.json</c>.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class Trellis2RealWeightsFactAttribute : FactAttribute
{
    public Trellis2RealWeightsFactAttribute(string? requireEnvironmentVariable = null)
    {
        if (FindFolder() is null)
        {
            Skip = "SKIPPED: Models/3D/trellis2 (pipeline.json and the flow checkpoints) not found; real-weight checks did not run.";
            return;
        }

        if (requireEnvironmentVariable is not null && Environment.GetEnvironmentVariable(requireEnvironmentVariable) != "1")
            Skip = $"SKIPPED: set {requireEnvironmentVariable}=1 to run this long real-weight run.";
    }

    /// <summary>The <c>3D/trellis2</c> folder, or null when it is not installed.</summary>
    public static string? FindFolder()
    {
        string? overridden = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS");
        if (!string.IsNullOrWhiteSpace(overridden) && File.Exists(Path.Combine(overridden, "3D", "trellis2", "pipeline.json")))
            return Path.Combine(overridden, "3D", "trellis2");
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "Models", "3D", "trellis2");
            if (File.Exists(Path.Combine(candidate, "pipeline.json")))
                return candidate;
            dir = dir.Parent;
        }

        return null;
    }
}
