using HartsyInference.Core.Runtime;

namespace HartsyInference.Engine.Planning;

/// <summary>One filesystem identity contract for planned video artifacts and their caches.</summary>
internal static class VideoArtifactPath
{
    /// <summary>Path equality matching the host filesystem convention.</summary>
    internal static StringComparer Comparer => FileSystemPathIdentity.Comparer;

    /// <summary>Path comparison matching <see cref="Comparer"/>.</summary>
    internal static StringComparison Comparison => FileSystemPathIdentity.Comparison;

    /// <summary>Stable text identity for cache keys, including case folding where path equality ignores case.</summary>
    internal static string Identity(string path)
    {
        return FileSystemPathIdentity.Identity(path);
    }

    /// <summary>Returns an absolute path with symbolic links in both parent directories and the final entry resolved.</summary>
    internal static string Canonicalize(string path)
    {
        return FileSystemPathIdentity.Canonicalize(path);
    }
}
