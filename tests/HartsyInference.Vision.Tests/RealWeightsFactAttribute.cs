namespace HartsyInference.Vision.Tests;

/// <summary>A <see cref="FactAttribute"/> that is reported as <i>skipped</i> (never passed) when any named environment
/// variable is unset or does not point at an existing file. xunit 2.x has no runtime skip, so the skip is set at
/// discovery, which is when xunit reads it.</summary>
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
                Skip = $"SKIPPED: {name} is unset or not an existing file; real-weight parity did not run.";
                return;
            }
        }
    }
}
