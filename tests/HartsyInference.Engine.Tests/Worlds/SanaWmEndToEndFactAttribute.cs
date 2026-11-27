namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>A <see cref="FactAttribute"/> for the SANA-WM end-to-end GPU run. It is reported as skipped at discovery unless
/// <c>HARTSYINFERENCE_MODELS_DIR</c> holds the SANA-WM release under <c>World/SanaWm</c> and <c>SANAWM_REFERENCE_RUN_DIR</c>
/// names an upstream run directory (<c>A_demo_pose</c>). With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is never skipped.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmEndToEndFactAttribute : FactAttribute
{
    /// <summary>Names the upstream run directory (index.json, image.safetensors, frames_png).</summary>
    public const string ReferenceEnvironmentVariable = "SANAWM_REFERENCE_RUN_DIR";

    /// <summary>Names the models root that holds <c>World/SanaWm</c>.</summary>
    public const string ModelsDirEnvironmentVariable = "HARTSYINFERENCE_MODELS_DIR";

    private const string RequireEnvironmentVariable = "HARTSY_REQUIRE_REAL_WEIGHTS";

    public SanaWmEndToEndFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(RequireEnvironmentVariable) == "1")
            return;
        string? reference = Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable);
        bool hasReference = !string.IsNullOrWhiteSpace(reference) && File.Exists(Path.Combine(reference, "image.safetensors"))
            && Directory.Exists(Path.Combine(reference, "frames_png"));
        if (!hasReference || DitPathOrNull() is null)
            Skip = $"SKIPPED: needs {ModelsDirEnvironmentVariable} (World/SanaWm release) and {ReferenceEnvironmentVariable} (an upstream run).";
    }

    /// <summary>The upstream run directory.</summary>
    public static string ReferenceDirectory() =>
        Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable)
        ?? throw new InvalidOperationException($"{ReferenceEnvironmentVariable} must name the upstream run directory.");

    /// <summary>The DiT checkpoint under the models root.</summary>
    public static string DitPath() =>
        DitPathOrNull() ?? throw new InvalidOperationException($"{ModelsDirEnvironmentVariable} must hold World/SanaWm/dit.");

    private static string? DitPathOrNull()
    {
        string? root = Environment.GetEnvironmentVariable(ModelsDirEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(root))
            return null;
        string path = Path.Combine(root, "World", "SanaWm", "dit", "sana_wm_1600m_720p.safetensors");
        return File.Exists(path) ? path : null;
    }
}
