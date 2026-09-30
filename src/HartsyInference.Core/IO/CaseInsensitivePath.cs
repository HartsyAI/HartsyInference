using System.Collections.Concurrent;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.IO;

/// <summary>Resolves a path under a root whose folder and file names may differ in case from the caller's spelling, so
/// a lookup for <c>LLM/qwen3</c> finds another tool's <c>llm/qwen3</c> on a case-sensitive filesystem.</summary>
/// <remarks>An existing exact path is returned exactly as <see cref="Path.Combine(string, string)"/> builds it, which
/// is always the case on case-insensitive filesystems (Windows, macOS). Otherwise each segment keeps its spelling when
/// that exists, else takes the one sibling of the right kind that matches ignoring case. The first segment with no such
/// sibling, or with two or more (ambiguous: logged once, never picked arbitrarily), keeps its spelling together with
/// every segment after it, so a path that is about to be created lands inside the folders that already exist. Only a
/// missing segment costs a listing of its parent directory; nothing is scanned recursively.</remarks>
public static class CaseInsensitivePath
{
    private static readonly char[] _separators = [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar];

    /// <summary>Ambiguous lookups already reported, keyed by the path as spelled, so a repeated lookup logs once.</summary>
    private static readonly ConcurrentDictionary<string, byte> _reportedAmbiguities = new(StringComparer.Ordinal);

    /// <summary>The file at <paramref name="relativePath"/> under <paramref name="root"/>, or the path it would be
    /// created at when no file matches.</summary>
    public static string ResolveFile(string root, string relativePath) => Resolve(root, relativePath, EntryKind.File);

    /// <summary>The directory at <paramref name="relativePath"/> under <paramref name="root"/>, or the path it would
    /// be created at when no directory matches.</summary>
    public static string ResolveDirectory(string root, string relativePath) =>
        Resolve(root, relativePath, EntryKind.Directory);

    /// <summary>The file or directory at <paramref name="relativePath"/> under <paramref name="root"/>, for a caller
    /// that accepts either; the path as spelled when neither matches.</summary>
    public static string ResolveEntry(string root, string relativePath) => Resolve(root, relativePath, EntryKind.Any);

    private static string Resolve(string root, string relativePath, EntryKind kind)
    {
        ArgumentNullException.ThrowIfNull(root);
        ArgumentNullException.ThrowIfNull(relativePath);
        string exact = Path.Combine(root, relativePath);
        if (Path.IsPathRooted(relativePath) || Exists(exact, kind))
        {
            return exact;
        }
        string[] segments = relativePath.Split(_separators, StringSplitOptions.RemoveEmptyEntries);
        string current = root;
        bool substituted = false;
        for (int i = 0; i < segments.Length; i++)
        {
            EntryKind wanted = i == segments.Length - 1 ? kind : EntryKind.Directory;
            string spelled = Path.Combine(current, segments[i]);
            if (Exists(spelled, wanted))
            {
                current = spelled;
                continue;
            }
            string? match = MatchIgnoringCase(current, segments[i], wanted);
            if (match is null)
            {
                return substituted
                    ? Path.Combine(current, string.Join(Path.DirectorySeparatorChar, segments, i, segments.Length - i))
                    : exact;
            }
            current = match;
            substituted = true;
        }
        return substituted ? current : exact;
    }

    /// <summary>The single entry of <paramref name="directory"/> of the wanted kind whose name equals
    /// <paramref name="name"/> ignoring case, or null when there is none or more than one.</summary>
    private static string? MatchIgnoringCase(string directory, string name, EntryKind kind)
    {
        if (!Directory.Exists(directory))
        {
            return null;
        }
        List<string>? matches = null;
        try
        {
            foreach (string entry in Directory.EnumerateFileSystemEntries(directory))
            {
                // Exists follows symlinks, which a store may use for its model folders; enumeration attributes do not.
                if (string.Equals(Path.GetFileName(entry), name, StringComparison.OrdinalIgnoreCase) && Exists(entry, kind))
                {
                    (matches ??= []).Add(entry);
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Debug($"[Paths] Could not list '{directory}' to match '{name}' ignoring case: {ex.Message}");
            return null;
        }
        if (matches is null)
        {
            return null;
        }
        if (matches.Count == 1)
        {
            return matches[0];
        }
        string spelled = Path.Combine(directory, name);
        if (_reportedAmbiguities.TryAdd(spelled, 0))
        {
            matches.Sort(StringComparer.Ordinal);
            Logs.Warning($"[Paths] '{name}' in '{directory}' matches {matches.Count} entries that differ only in case "
                + $"({string.Join(", ", matches)}); none is chosen, so '{spelled}' is used as spelled. "
                + "Merge them into one.");
        }
        return null;
    }

    private static bool Exists(string path, EntryKind kind) => kind switch
    {
        EntryKind.File => File.Exists(path),
        EntryKind.Directory => Directory.Exists(path),
        _ => File.Exists(path) || Directory.Exists(path),
    };

    private enum EntryKind
    {
        File,
        Directory,
        Any,
    }
}
