namespace HartsyInference.Core.Runtime;

/// <summary>Canonical filesystem identity shared by components that bind or overwrite local artifacts.</summary>
internal static class FileSystemPathIdentity
{
    internal static StringComparer Comparer { get; } = OperatingSystem.IsWindows()
        ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    internal static StringComparison Comparison { get; } = OperatingSystem.IsWindows()
        ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    internal static string Identity(string path)
    {
        string canonical = Canonicalize(path);
        return OperatingSystem.IsWindows() ? canonical.ToUpperInvariant() : canonical;
    }

    internal static bool SamePath(string left, string right) =>
        Comparer.Equals(Canonicalize(left), Canonicalize(right));

    /// <summary>Returns an absolute path with symbolic links in parent directories and the final entry resolved.</summary>
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
