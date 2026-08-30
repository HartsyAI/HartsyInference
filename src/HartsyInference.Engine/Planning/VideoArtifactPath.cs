namespace HartsyInference.Engine.Planning;

/// <summary>One filesystem identity contract for planned video artifacts and their caches.</summary>
internal static class VideoArtifactPath
{
    /// <summary>Path equality matching the host filesystem convention.</summary>
    internal static StringComparer Comparer { get; } = OperatingSystem.IsWindows()
        ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    /// <summary>Path comparison matching <see cref="Comparer"/>.</summary>
    internal static StringComparison Comparison { get; } = OperatingSystem.IsWindows()
        ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>Stable text identity for cache keys, including case folding where path equality ignores case.</summary>
    internal static string Identity(string path)
    {
        string canonical = Canonicalize(path);
        return OperatingSystem.IsWindows() ? canonical.ToUpperInvariant() : canonical;
    }

    /// <summary>Returns an absolute path with symbolic links in both parent directories and the final entry resolved.</summary>
    internal static string Canonicalize(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        string? root = Path.GetPathRoot(fullPath);
        if (string.IsNullOrEmpty(root))
        {
            return fullPath;
        }

        string current = root;
        string relative = Path.GetRelativePath(root, fullPath);
        foreach (string segment in relative.Split(
                     [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
                     StringSplitOptions.RemoveEmptyEntries))
        {
            string candidate = Path.Combine(current, segment);
            FileSystemInfo? target = Directory.Exists(candidate)
                ? Directory.ResolveLinkTarget(candidate, returnFinalTarget: true)
                : File.Exists(candidate) ? File.ResolveLinkTarget(candidate, returnFinalTarget: true) : null;
            current = target?.FullName ?? candidate;
        }
        return Path.GetFullPath(current);
    }
}
