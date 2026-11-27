using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A fact reported as <i>skipped</i> (never passed) when a named environment variable is unset or not an existing file.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class RealWeightsFactAttribute : FactAttribute
{
    public RealWeightsFactAttribute(params string[] environmentVariables)
    {
        foreach (string name in environmentVariables)
        {
            string? path = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                Skip = $"SKIPPED: {name} is unset or not an existing file; real-weight check did not run.";
                return;
            }
        }
    }
}
