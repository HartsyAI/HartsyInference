using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A <see cref="FactAttribute"/> for checks against an upstream SANA-WM run. It is skipped unless
/// <c>HARTSYINFERENCE_MODELS_DIR</c> holds the DiT and <c>SANAWM_REFERENCE_RUN_DIR</c> holds the run's dumps; with
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it always runs, so missing files fail.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmBlockReferenceFactAttribute : FactAttribute
{
    public const string ReferenceEnvironmentVariable = "SANAWM_REFERENCE_RUN_DIR";

    public SanaWmBlockReferenceFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(SanaWmRealWeightsFactAttribute.RequireEnvironmentVariable) == "1")
            return;
        string? reference = Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable);
        bool hasReference = !string.IsNullOrWhiteSpace(reference) && File.Exists(Path.Combine(reference, "units.safetensors"));
        bool hasDit = Environment.GetEnvironmentVariable(SanaWmRealWeightsFactAttribute.ModelsDirEnvironmentVariable) is { Length: > 0 } root
            && File.Exists(Path.Combine(root, "World", "SanaWm", "dit", "sana_wm_1600m_720p.safetensors"));
        if (!hasDit || !hasReference)
            Skip = $"SKIPPED: needs HARTSYINFERENCE_MODELS_DIR (the DiT) and {ReferenceEnvironmentVariable} (the upstream run dumps).";
    }

    /// <summary>The upstream run directory, or throws when the variable is unset.</summary>
    public static string ReferenceDirectory() =>
        Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable)
        ?? throw new InvalidOperationException($"{ReferenceEnvironmentVariable} must name the upstream run directory.");
}
