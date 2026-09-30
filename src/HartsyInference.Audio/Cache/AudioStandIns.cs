using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Cache;

/// <summary>Lets a converted checkpoint stand in for the upstream files it was made from. A Hartsy artifact names,
/// in its <c>hartsy.stands_in_for</c> metadata, the paths under the audio models root that the engine's loaders
/// open (<c>tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin</c>, <c>music/acestep/acestep-v15-turbo.safetensors</c>);
/// <see cref="Sync"/> links the artifact into each of those paths that is missing, so every loader finds it where it
/// already looks, whatever it is called on disk and whichever of the engine's acquisition paths the family uses.
/// <para>Links are hard links (no copy, and a size check sees the real file); a symbolic link is the fallback when the
/// target is on another filesystem. Links are recorded in <c>.hartsy-standins.json</c> at the root, so removing the
/// artifact removes its links on the next sync instead of leaving them to be mistaken for an upstream download. An
/// existing file is never replaced: a real upstream copy wins over a stand-in.</para></summary>
public static partial class AudioStandIns
{
    /// <summary>Metadata key listing the audio-root-relative paths an artifact stands in for, separated by <c>;</c>.</summary>
    public const string MetadataKey = "hartsy.stands_in_for";

    private const string ManifestName = ".hartsy-standins.json";
    private static readonly TimeSpan ResyncInterval = TimeSpan.FromSeconds(2);
    private static readonly object _lock = new();
    private static readonly Dictionary<string, DateTime> _lastSync = new(StringComparer.Ordinal);
    private static readonly Dictionary<(string Path, long Length, DateTime Written), string[]> _declared = [];

    /// <summary>Links every artifact under <paramref name="root"/> into its declared paths, once per process; later
    /// calls return immediately. Loaders that open a fixed path call this before looking.</summary>
    public static void EnsureSynced(string root)
    {
        lock (_lock)
        {
            if (_lastSync.ContainsKey(Path.GetFullPath(root)))
            {
                return;
            }
        }
        Sync(root);
    }

    /// <summary>Rescans <paramref name="root"/> unless it was scanned in the last two seconds: for a lookup that just
    /// missed, since an artifact may have been installed while the engine was running.</summary>
    public static void Resync(string root)
    {
        lock (_lock)
        {
            if (_lastSync.TryGetValue(Path.GetFullPath(root), out DateTime last) && DateTime.UtcNow - last < ResyncInterval)
            {
                return;
            }
        }
        Sync(root);
    }

    /// <summary>Removes links whose artifact is gone, then links each artifact into its declared paths that do not
    /// exist. Returns the number of links created. Never throws: a failure is logged and the loader's own
    /// not-found handling takes over.</summary>
    public static int Sync(string root)
    {
        string fullRoot = Path.GetFullPath(root);
        lock (_lock)
        {
            _lastSync[fullRoot] = DateTime.UtcNow;
            try
            {
                return SyncLocked(fullRoot);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
            {
                Logs.Warning($"[AudioCache] Could not link converted checkpoints under '{fullRoot}': {ex.Message}");
                return 0;
            }
        }
    }

    /// <summary>The paths <paramref name="safetensorsPath"/> declares it stands in for, or empty. Paths that are rooted
    /// or climb out of the root are dropped: a file must not be able to write outside the models tree.</summary>
    public static IReadOnlyList<string> ReadDeclared(string safetensorsPath)
    {
        string? value = ReadMetadataValue(safetensorsPath, MetadataKey);
        if (string.IsNullOrWhiteSpace(value))
        {
            return [];
        }
        List<string> paths = [];
        foreach (string raw in value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string relative = raw.Replace('\\', '/');
            if (Path.IsPathRooted(relative) || relative.Split('/').Any(s => s is ".." or "." or ""))
            {
                Logs.Warning($"[AudioCache] '{Path.GetFileName(safetensorsPath)}' declares '{raw}', which is not a path inside the models root; ignored.");
                continue;
            }
            paths.Add(relative);
        }
        return paths;
    }

    private static int SyncLocked(string root)
    {
        if (!Directory.Exists(root))
        {
            return 0;
        }
        string manifestPath = Path.Combine(root, ManifestName);
        Dictionary<string, LinkRecord> links = ReadManifest(manifestPath, root, out bool changed);
        foreach ((string link, LinkRecord record) in links.ToList())
        {
            string linkPath = Path.Combine(root, link);
            string targetPath = Path.Combine(root, record.Target);
            if (!IsOurLink(linkPath, targetPath, record))
            {
                // Removed, or replaced by a real file: no longer ours to keep or delete.
                links.Remove(link);
                changed = true;
                continue;
            }
            if (File.Exists(targetPath) && (record.Symbolic || MatchesRecord(new FileInfo(targetPath), record)))
            {
                continue;
            }
            // The artifact was removed, or replaced by a newer copy the hard link does not follow: drop the old link
            // (the scan below relinks a replaced artifact).
            TryDelete(linkPath);
            links.Remove(link);
            changed = true;
        }
        HashSet<string> linkSet = new(links.Keys.Select(l => Path.Combine(root, l)), StringComparer.Ordinal);
        int created = 0;
        foreach (string file in Directory.EnumerateFiles(root, "*.safetensors", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true }))
        {
            if (linkSet.Contains(file))
            {
                continue;
            }
            foreach (string relative in Declared(file))
            {
                string destination = Path.Combine(root, relative);
                if (File.Exists(destination) || Directory.Exists(destination))
                {
                    continue;
                }
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                bool hard = TryHardLink(file, destination);
                if (!hard)
                {
                    File.CreateSymbolicLink(destination, file);
                }
                FileInfo target = new(file);
                links[relative] = new LinkRecord(Path.GetRelativePath(root, file).Replace('\\', '/'), target.Length, target.LastWriteTimeUtc.Ticks, !hard);
                linkSet.Add(destination);
                changed = true;
                created++;
                Logs.Info($"[AudioCache] {relative} -> {links[relative].Target} ({(hard ? "hard link" : "symlink")}).");
            }
        }
        if (changed)
        {
            WriteManifest(manifestPath, links);
        }
        return created;
    }

    /// <summary>One link this class made: its artifact and, for a hard link, the size and write time the two shared
    /// when it was made. There is no portable inode API, so those stand in for identity: a hard link keeps them while
    /// a replaced artifact or a file dropped in over the link does not.</summary>
    private sealed record LinkRecord(string Target, long Length, long WriteTicks, bool Symbolic);

    private static bool MatchesRecord(FileInfo info, LinkRecord record) =>
        info.Exists && info.Length == record.Length && info.LastWriteTimeUtc.Ticks == record.WriteTicks;

    private static bool IsOurLink(string linkPath, string targetPath, LinkRecord record)
    {
        FileInfo info = new(linkPath);
        if (record.Symbolic)
        {
            return info.LinkTarget is { } pointsAt && Path.GetFullPath(pointsAt, Path.GetDirectoryName(linkPath)!) == Path.GetFullPath(targetPath);
        }
        return info.LinkTarget is null && MatchesRecord(info, record);
    }

    private static string[] Declared(string file)
    {
        FileInfo info = new(file);
        (string, long, DateTime) key = (file, info.Length, info.LastWriteTimeUtc);
        if (!_declared.TryGetValue(key, out string[]? paths))
        {
            try
            {
                paths = [.. ReadDeclared(file)];
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or InvalidDataException)
            {
                Logs.Debug($"[AudioCache] Skipping unreadable '{file}': {ex.Message}");
                paths = [];
            }
            _declared[key] = paths;
        }
        return paths;
    }

    /// <summary>Reads one <c>__metadata__</c> value from a safetensors header without touching the tensors.</summary>
    private static string? ReadMetadataValue(string path, string key)
    {
        using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        Span<byte> head = stackalloc byte[8];
        if (stream.Read(head) < 8)
        {
            return null;
        }
        long length = BitConverter.ToInt64(head);
        if (length <= 1 || length > Math.Min(stream.Length - 8, 100L << 20))
        {
            throw new InvalidDataException("not a safetensors header");
        }
        byte[] header = new byte[length];
        stream.ReadExactly(header);
        using JsonDocument doc = JsonDocument.Parse(header);
        return doc.RootElement.TryGetProperty("__metadata__", out JsonElement meta) && meta.ValueKind == JsonValueKind.Object
            && meta.TryGetProperty(key, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    /// <param name="upgraded">True when the file was in the early targets-only format, so it must be rewritten.</param>
    private static Dictionary<string, LinkRecord> ReadManifest(string path, string root, out bool upgraded)
    {
        Dictionary<string, LinkRecord> links = new(StringComparer.Ordinal);
        upgraded = false;
        if (!File.Exists(path))
        {
            return links;
        }
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
        foreach (JsonProperty entry in doc.RootElement.EnumerateObject())
        {
            if (entry.Value.ValueKind == JsonValueKind.Object)
            {
                JsonElement v = entry.Value;
                links[entry.Name] = new LinkRecord(v.GetProperty("target").GetString()!, v.GetProperty("length").GetInt64(),
                    v.GetProperty("writeTicks").GetInt64(), v.TryGetProperty("symbolic", out JsonElement sym) && sym.GetBoolean());
            }
            else if (entry.Value.GetString() is { } target)
            {
                upgraded = true;
                // An early manifest recorded only the target: adopt the link only if it still matches the artifact.
                FileInfo linkInfo = new(Path.Combine(root, entry.Name));
                FileInfo targetInfo = new(Path.Combine(root, target));
                if (linkInfo.Exists && targetInfo.Exists && linkInfo.Length == targetInfo.Length && linkInfo.LastWriteTimeUtc == targetInfo.LastWriteTimeUtc)
                {
                    links[entry.Name] = new LinkRecord(target, targetInfo.Length, targetInfo.LastWriteTimeUtc.Ticks, linkInfo.LinkTarget is not null);
                }
            }
        }
        return links;
    }

    private static void WriteManifest(string path, Dictionary<string, LinkRecord> links)
    {
        string temp = path + ".tmp";
        SortedDictionary<string, object> entries = new(StringComparer.Ordinal);
        foreach ((string link, LinkRecord r) in links)
        {
            entries[link] = new { target = r.Target, length = r.Length, writeTicks = r.WriteTicks, symbolic = r.Symbolic };
        }
        File.WriteAllText(temp, JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temp, path, overwrite: true);
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path) || new FileInfo(path).LinkTarget is not null)
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Debug($"[AudioCache] Could not remove stale link '{path}': {ex.Message}");
        }
    }

    private static bool TryHardLink(string existing, string link)
    {
        try
        {
            return OperatingSystem.IsWindows() ? CreateHardLinkW(link, existing, IntPtr.Zero) : LinkUnix(existing, link) == 0;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [LibraryImport("libc", EntryPoint = "link", SetLastError = true, StringMarshalling = StringMarshalling.Utf8)]
    private static partial int LinkUnix(string existing, string link);

    [LibraryImport("kernel32.dll", EntryPoint = "CreateHardLinkW", SetLastError = true, StringMarshalling = StringMarshalling.Utf16)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CreateHardLinkW(string link, string existing, IntPtr securityAttributes);
}
