namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) when the Pixal3D files are not installed. The folder is found from
/// <c>HARTSYINFERENCE_MODELS</c>, else by walking up from the test binary to a <c>Models/3D/pixal3d/pipeline.json</c>.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class Pixal3DRealWeightsFactAttribute : FactAttribute
{
    public Pixal3DRealWeightsFactAttribute()
    {
        if (FindFolder() is null)
            Skip = "SKIPPED: Models/3D/pixal3d (pipeline.json and the flow checkpoints) not found; real-weight checks did not run.";
    }

    /// <summary>The <c>3D/pixal3d</c> folder, or null when it is not installed.</summary>
    public static string? FindFolder()
    {
        string? overridden = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS");
        if (!string.IsNullOrWhiteSpace(overridden) && File.Exists(Path.Combine(overridden, "3D", "pixal3d", "pipeline.json")))
            return Path.Combine(overridden, "3D", "pixal3d");
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "Models", "3D", "pixal3d");
            if (File.Exists(Path.Combine(candidate, "pipeline.json")))
                return candidate;
            dir = dir.Parent;
        }

        return null;
    }
}
