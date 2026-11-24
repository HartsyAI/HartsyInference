using Xunit;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

/// <summary>A <see cref="FactAttribute"/> reported as <i>skipped</i> (never passed) when a named environment variable is unset or names
/// no existing file or folder. With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it runs anyway, so a missing resource fails the test.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class RealWeightsFactAttribute : FactAttribute
{
    public RealWeightsFactAttribute(params string[] environmentVariables)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1")
        {
            return;
        }
        foreach (string name in environmentVariables)
        {
            string? path = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(path) || !(File.Exists(path) || Directory.Exists(path)))
            {
                Skip = $"SKIPPED: {name} is unset or names no existing file or folder; real-weight check did not run.";
                return;
            }
        }
    }
}
