namespace HartsyInference.Tests.Common;

/// <summary>Checkpoint gate for real-weight tests. By default a missing file logs a SKIPPED line and the test
/// returns early (today's behavior, safe for CI machines without weights); with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>
/// — set by <c>tests/run-multigpu-campaign.sh</c> — a missing file throws instead, so a verification campaign can
/// never report green on a silently-skipped test.</summary>
public static class RealWeightGate
{
    /// <summary>Environment variable that converts silent skips into hard failures.</summary>
    public const string RequireEnvVar = "HARTSY_REQUIRE_REAL_WEIGHTS";

    /// <summary>True when every path exists. On a miss, either throws (require mode) or logs "SKIPPED: ..." via <paramref name="log"/> and returns false — callers do <c>if (!RealWeightGate.Require(_output.WriteLine, ...)) return;</c>.</summary>
    public static bool Require(Action<string> log, params string[] paths)
    {
        ArgumentNullException.ThrowIfNull(log);
        List<string>? missing = null;
        foreach (string path in paths)
        {
            if (!File.Exists(path) && !Directory.Exists(path))
            {
                (missing ??= []).Add(path);
            }
        }
        if (missing is null)
        {
            return true;
        }

        string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
        if (Environment.GetEnvironmentVariable(RequireEnvVar) == "1")
        {
            throw new InvalidOperationException(
                $"{RequireEnvVar}=1 but {message}. Download the checkpoint(s) or unset the variable to allow skips.");
        }
        log($"SKIPPED: {message}");
        return false;
    }
}
