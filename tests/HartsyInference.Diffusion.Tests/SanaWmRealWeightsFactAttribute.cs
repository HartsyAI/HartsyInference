using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A <see cref="FactAttribute"/> for SANA-WM real-weight checks. It is reported as skipped at discovery when
/// <c>HARTSYINFERENCE_MODELS_DIR</c> does not hold the DiT file. With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is not skipped,
/// and a missing file fails the test instead.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmRealWeightsFactAttribute : FactAttribute
{
    public const string ModelsDirEnvironmentVariable = "HARTSYINFERENCE_MODELS_DIR";
    public const string RequireEnvironmentVariable = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const string DitRelativePath = "World/SanaWm/dit/sana_wm_1600m_720p.safetensors";

    public SanaWmRealWeightsFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(RequireEnvironmentVariable) == "1")
            return;
        if (ResolveDitPathOrNull() is null)
            Skip = $"SKIPPED: {ModelsDirEnvironmentVariable} does not hold {DitRelativePath}; "
                + "the real-header check did not run.";
    }

    /// <summary>Returns the DiT file under <c>HARTSYINFERENCE_MODELS_DIR</c>, or throws when it is not there.</summary>
    public static string ResolveDitPath() =>
        ResolveDitPathOrNull()
        ?? throw new InvalidOperationException(
            $"{ModelsDirEnvironmentVariable} must hold {DitRelativePath}; the real-weight check cannot run.");

    private static string? ResolveDitPathOrNull()
    {
        string? root = Environment.GetEnvironmentVariable(ModelsDirEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(root))
            return null;
        string path = Path.Combine(root, DitRelativePath.Replace('/', Path.DirectorySeparatorChar));
        return File.Exists(path) ? path : null;
    }
}
