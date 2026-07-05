namespace HartsyInference.Cli.Infra;

/// <summary>Resolves the well-known roots (repo, models, output) once, honoring env overrides, so commands do not
/// re-implement the walk-up-to-the-solution logic that was copied across the sample CLIs.</summary>
public static class RepoPaths
{
    /// <summary>The repository root: the <c>HARTSYINFERENCE_REPO_ROOT</c> override, else the nearest ancestor of the
    /// executable that contains <c>HartsyInference.sln</c>, else the current directory.</summary>
    public static string RepoRoot()
    {
        string? overridden = Environment.GetEnvironmentVariable("HARTSYINFERENCE_REPO_ROOT");
        if (!string.IsNullOrEmpty(overridden) && Directory.Exists(overridden))
            return Path.GetFullPath(overridden);

        string current = AppContext.BaseDirectory;
        for (int depth = 0; depth < 12; depth++)
        {
            if (File.Exists(Path.Combine(current, "HartsyInference.sln")))
                return current;
            string? parent = Path.GetDirectoryName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.IsNullOrEmpty(parent) || parent == current)
                break;
            current = parent;
        }
        return Environment.CurrentDirectory;
    }

    /// <summary>The models root: the <c>HARTSYINFERENCE_MODELS</c> override, else <c>&lt;repo&gt;/Models</c>.</summary>
    public static string ModelsRoot() =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS") is { Length: > 0 } m
            ? Path.GetFullPath(m)
            : Path.Combine(RepoRoot(), "Models");

    /// <summary>The output root: the <c>HARTSYINFERENCE_OUTPUT</c> override, else <c>&lt;repo&gt;/Output</c>.</summary>
    public static string OutputRoot() =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_OUTPUT") is { Length: > 0 } o
            ? Path.GetFullPath(o)
            : Path.Combine(RepoRoot(), "Output");
}
