using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A fact reported as <i>skipped</i> (never passed) when a named environment variable is unset or not an existing file.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class RealWeightsFactAttribute : FactAttribute
{
    public RealWeightsFactAttribute(params string[] environmentVariables)
    {
        Skip = MissingFileSkip(environmentVariables);
    }

    /// <summary>The skip reason when a named variable is unset or not an existing file; null when all are present.</summary>
    internal static string? MissingFileSkip(IEnumerable<string> environmentVariables)
    {
        foreach (string name in environmentVariables)
        {
            string? path = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                return $"SKIPPED: {name} is unset or not an existing file; real-weight check did not run.";
            }
        }
        return null;
    }
}

/// <summary>A real-weight fact that is also skipped when no CUDA device opens: the skip reason names whichever is missing.</summary>
internal sealed class CudaRealWeightsFactAttribute : FactAttribute
{
    public CudaRealWeightsFactAttribute(params string[] environmentVariables)
    {
        Skip = GpuDevices.CudaFailure is { } failure
            ? $"SKIPPED: no usable CUDA device ({failure})."
            : RealWeightsFactAttribute.MissingFileSkip(environmentVariables);
    }
}
