using Xunit;

namespace HartsyInference.World.Tests.Camera;

/// <summary>A <see cref="FactAttribute"/> for checks against the upstream SANA-WM reference dumps. It is reported as skipped
/// when <c>SANAWM_REFERENCE_RUNS_DIR</c> does not name the directory holding the <c>A_demo_pose</c> and <c>B_action161</c>
/// runs. With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is not skipped, and a missing directory fails the test.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmReferenceRunsFactAttribute : FactAttribute
{
    /// <summary>Names the directory that holds the reference runs.</summary>
    public const string RunsEnvironmentVariable = "SANAWM_REFERENCE_RUNS_DIR";

    private const string RequireEnvironmentVariable = "HARTSY_REQUIRE_REAL_WEIGHTS";

    public SanaWmReferenceRunsFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(RequireEnvironmentVariable) == "1")
            return;
        if (RunsDirectoryOrNull() is null)
            Skip = $"SKIPPED: {RunsEnvironmentVariable} does not name the SANA-WM reference runs directory.";
    }

    /// <summary>The directory of run <paramref name="name"/>, or a failure naming the variable to set.</summary>
    public static string Run(string name)
    {
        string root = RunsDirectoryOrNull()
            ?? throw new InvalidOperationException($"{RunsEnvironmentVariable} must name the SANA-WM reference runs directory.");
        string run = Path.Combine(root, name);
        return Directory.Exists(run) ? run : throw new DirectoryNotFoundException($"Reference run '{name}' is not under {root}.");
    }

    private static string? RunsDirectoryOrNull()
    {
        string? root = Environment.GetEnvironmentVariable(RunsEnvironmentVariable);
        return string.IsNullOrWhiteSpace(root) || !Directory.Exists(Path.Combine(root, "A_demo_pose")) ? null : root;
    }
}
