using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>A fact reported as skipped, never passed, when the named environment variable is unset or not an existing folder.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
internal sealed class DuoRealWeightsFactAttribute : FactAttribute
{
    public DuoRealWeightsFactAttribute(string environmentVariable)
    {
        string? path = Environment.GetEnvironmentVariable(environmentVariable);
        if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            Skip = $"SKIPPED: {environmentVariable} is unset or not an existing folder; real-weight check did not run.";
    }
}
