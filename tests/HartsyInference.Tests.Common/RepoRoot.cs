namespace HartsyInference.Tests.Common;

/// <summary>Resolves the repo root directory once per process. OS-agnostic: walks up from the test assembly's bin folder looking for the HartsyInference.sln marker.</summary>
public static class RepoRoot
{
    /// <summary>Absolute path to the repo root. Override with HARTSYINFERENCE_REPO_ROOT.</summary>
    public static string Path { get; } = Resolve();

    private static string Resolve()
    {
        string? overridden = Environment.GetEnvironmentVariable("HARTSYINFERENCE_REPO_ROOT");
        if (!string.IsNullOrEmpty(overridden) && Directory.Exists(overridden))
            return System.IO.Path.GetFullPath(overridden);

        string current = AppContext.BaseDirectory;
        for (int depth = 0; depth < 12; depth++)
        {
            if (File.Exists(System.IO.Path.Combine(current, "HartsyInference.sln")))
                return current;
            string? parent = System.IO.Path.GetDirectoryName(current.TrimEnd(System.IO.Path.DirectorySeparatorChar, System.IO.Path.AltDirectorySeparatorChar));
            if (string.IsNullOrEmpty(parent) || parent == current)
                break;
            current = parent;
        }

        throw new InvalidOperationException(
            $"Could not locate HartsyInference.sln walking up from '{AppContext.BaseDirectory}'. " +
            "Set HARTSYINFERENCE_REPO_ROOT to override.");
    }
}
